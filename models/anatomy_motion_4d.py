"""AnatomyMotion4DNet: local anatomy, spatial transitions, diagnostic phase."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.interval_selector import (
    IntervalEndpointHead,
    interval_diagnostics,
    interval_distribution,
    motion_phase_weights,
    phase_membership,
)
from models.raw_motion import (
    RawDifferenceEncoder,
    RawFeatureMotionFusion,
    RawMotionProjection,
    RawToMotionChannels,
    valid_raw_motion_pairs,
)
from models.resnet3d_features import ResNet3D18Features
from models.spatial_attention import SpatialAttentionPool3d, SpatialGapPool3d, sector_at_resolution
from models.spatial_transition import (
    SpatialTransitionEncoder,
    build_transition_input,
    transition_in_channels,
)
from models.deformable_temporal import CrossTemporalDeformAlign, TemporalDeformEncoder
from models.temporal import (
    ContinuousTimeEncoding,
    TemporalEncoder,
    map_motion_to_state,
    normalized_cardiac_time,
)


VARIANTS = ("M0", "M1", "M2", "M3", "M4")
LOCAL_LAYERS = ("layer2", "layer3")
POOLS = ("attention", "gap")
TRANSITION_REPRESENTATIONS = ("full", "pair", "difference")
# Classifier concat for the M3/M4 selector path. Default follows variant.
HEAD_FEATURES = (
    "phase_anatomy_phase_motion_global",  # M4 default
    "phase_anatomy_phase_motion",         # drop z_global
    "phase_motion_global",                # drop z_phase_anatomy
)
TEMPORAL_POOLS = ("phase", "uniform")
# Optional temporal plugins. Defaults keep the original MotionR_S path.
TEMPORAL_MIXERS = ("dense", "deform_attn")
MOTION_ALIGNS = ("linear", "cross_deform")


def _feature_dim(variant, d_model, head_features):
    if head_features is None:
        return {"M0": d_model, "M1": d_model, "M2": d_model * 2, "M3": d_model * 2, "M4": d_model * 3}[variant]
    widths = {
        "phase_anatomy_phase_motion_global": d_model * 3,
        "phase_anatomy_phase_motion": d_model * 2,
        "phase_motion_global": d_model * 2,
    }
    return widths[head_features]


def _assemble_head_features(z_phase_anatomy, z_phase_motion, z_global, head_features):
    if head_features == "phase_anatomy_phase_motion_global":
        return torch.cat([z_phase_anatomy, z_phase_motion, z_global], dim=-1)
    if head_features == "phase_anatomy_phase_motion":
        return torch.cat([z_phase_anatomy, z_phase_motion], dim=-1)
    if head_features == "phase_motion_global":
        return torch.cat([z_phase_motion, z_global], dim=-1)
    raise ValueError(f"unknown head_features {head_features}")


def _scatter(values, index, batch, steps):
    out = values.new_zeros(batch * steps, *values.shape[1:])
    if index.numel():
        out = out.index_copy(0, index, values)
    return out.view(batch, steps, *values.shape[1:])


def _masked_mean(tokens, mask):
    weight = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)


def _assert_prefix_mask(frame_mask):
    lengths = frame_mask.sum(dim=1)
    index = torch.arange(frame_mask.shape[1], device=frame_mask.device)
    expected = index.view(1, -1) < lengths.view(-1, 1)
    if not torch.equal(frame_mask, expected):
        raise ValueError("frame_mask must be True on a prefix and False only at the end")
    return lengths


class AnatomyMotion4DNet(nn.Module):
    """Binary MR network. Logits are returned without a sigmoid."""

    def __init__(
        self,
        variant="M4",
        local_layer="layer2",
        local_pooling="attention",
        transition_representation="full",
        transition_pooling="attention",
        d_model=256,
        nhead=4,
        temporal_layers=2,
        dim_feedforward=512,
        temporal_dropout=0.10,
        transition_dropout=0.10,
        selector_dropout=0.10,
        selector_hidden=128,
        classifier_dropout=0.20,
        classifier_hidden=256,
        attention_hidden=64,
        transition_hidden_channels=256,
        transition_output_channels=None,
        transition_attention_hidden=64,
        transition_attention_dim=None,
        fusion_dropout=0.10,
        fusion_dim=None,
        use_raw_difference=True,
        use_feature_transition=True,
        head_features=None,
        temporal_pool="phase",
        use_sector_mask=False,
        use_sector_mask_global=False,
        use_stem_frame_diff=False,
        stem_diff_mode="abs",
        temporal_mixer="dense",
        motion_align="linear",
        deform_n_points=4,
        checkpoint_encoder=True,
        encoder_chunk=8,
        transition_chunk=8,
        time_max_cycles=32.0,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if local_layer not in LOCAL_LAYERS:
            raise ValueError(f"local_layer must be one of {LOCAL_LAYERS}")
        if head_features is None:
            if variant == "M3":
                head_features = "phase_anatomy_phase_motion"
            elif variant == "M4":
                head_features = "phase_anatomy_phase_motion_global"
        if head_features is not None and head_features not in HEAD_FEATURES:
            raise ValueError(f"head_features must be one of {HEAD_FEATURES}")
        if head_features is not None and variant not in ("M3", "M4"):
            raise ValueError("head_features only applies to M3/M4")
        if temporal_pool not in TEMPORAL_POOLS:
            raise ValueError(f"temporal_pool must be one of {TEMPORAL_POOLS}")
        if temporal_pool == "uniform" and variant not in ("M3", "M4"):
            raise ValueError("temporal_pool=uniform only applies to M3/M4")
        if temporal_mixer not in TEMPORAL_MIXERS:
            raise ValueError(f"temporal_mixer must be one of {TEMPORAL_MIXERS}")
        if motion_align not in MOTION_ALIGNS:
            raise ValueError(f"motion_align must be one of {MOTION_ALIGNS}")
        if int(deform_n_points) < 1:
            raise ValueError("deform_n_points must be positive")
        if stem_diff_mode not in ("abs", "signed"):
            raise ValueError("stem_diff_mode must be 'abs' or 'signed'")
        if local_pooling not in POOLS or transition_pooling not in POOLS:
            raise ValueError(f"pooling must be one of {POOLS}")
        if transition_representation not in TRANSITION_REPRESENTATIONS:
            raise ValueError(f"transition_representation must be one of {TRANSITION_REPRESENTATIONS}")
        if encoder_chunk < 1 or transition_chunk < 1:
            raise ValueError("chunk sizes must be positive")
        if transition_output_channels is None:
            transition_output_channels = transition_hidden_channels
        if transition_attention_dim is None:
            transition_attention_dim = transition_output_channels
        if fusion_dim is None:
            fusion_dim = d_model

        self.variant = variant
        self.local_layer = local_layer
        self.local_pooling = local_pooling
        self.transition_representation = transition_representation
        self.transition_pooling = transition_pooling
        self.checkpoint_encoder = bool(checkpoint_encoder)
        self.encoder_chunk = int(encoder_chunk)
        self.transition_chunk = int(transition_chunk)
        self.d_model = int(d_model)
        self.use_raw_difference = bool(use_raw_difference)
        self.use_feature_transition = bool(use_feature_transition)
        self.head_features = head_features
        self.temporal_pool = temporal_pool
        self.use_sector_mask = bool(use_sector_mask)
        self.use_sector_mask_global = bool(use_sector_mask_global)
        self.use_stem_frame_diff = bool(use_stem_frame_diff)
        self.stem_diff_mode = stem_diff_mode
        self.temporal_mixer = temporal_mixer
        self.motion_align = motion_align
        self.deform_n_points = int(deform_n_points)

        self.encoder = ResNet3D18Features(in_channels=2 if self.use_stem_frame_diff else 1)
        self.local_channels = ResNet3D18Features.LAYER_CHANNELS[local_layer]
        self.global_channels = ResNet3D18Features.LAYER_CHANNELS["layer4"]

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(self.global_channels, d_model),
            nn.LayerNorm(d_model),
        )
        pool_cls = SpatialAttentionPool3d if local_pooling == "attention" else SpatialGapPool3d
        if local_pooling == "attention":
            self.local_pool = pool_cls(self.local_channels, attention_hidden, out_dim=d_model)
        else:
            self.local_pool = pool_cls(self.local_channels, out_dim=d_model)
        self.state_token = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.state_time = ContinuousTimeEncoding(d_model, max_cycles=time_max_cycles)
        # Keep the original module name on the default path so MotionR_S is unchanged.
        if temporal_mixer == "dense":
            self.state_transformer = TemporalEncoder(
                d_model, nhead, temporal_layers, dim_feedforward, temporal_dropout,
            )
            self.state_deform_encoder = None
        else:
            self.state_transformer = None
            self.state_deform_encoder = TemporalDeformEncoder(
                d_model, nhead, temporal_layers, dim_feedforward, temporal_dropout,
                n_points=self.deform_n_points,
            )

        in_channels = transition_in_channels(self.local_channels, transition_representation)
        self.transition_encoder = None
        if self.use_feature_transition:
            self.transition_encoder = SpatialTransitionEncoder(
                in_channels,
                hidden_channels=transition_hidden_channels,
                output_channels=transition_output_channels,
                dropout=transition_dropout,
            )
        self.raw_difference_encoder = RawDifferenceEncoder() if self.use_raw_difference else None
        self.raw_motion_proj = RawMotionProjection() if self.use_raw_difference else None
        feature_channels = int(transition_output_channels)
        if self.use_raw_difference and self.use_feature_transition:
            self.motion_fusion = RawFeatureMotionFusion(
                feature_channels + 128, self.d_model, dropout=transition_dropout,
            )
            self.raw_to_attention = None
            attention_in = self.d_model
        elif self.use_raw_difference:
            self.motion_fusion = None
            self.raw_to_attention = RawToMotionChannels(self.d_model)
            attention_in = self.d_model
        else:
            self.motion_fusion = None
            self.raw_to_attention = None
            attention_in = feature_channels
        pooled_dim = int(transition_attention_dim)
        pool_out = None if pooled_dim == attention_in else pooled_dim
        if transition_pooling == "attention":
            self.transition_pool = SpatialAttentionPool3d(
                attention_in, transition_attention_hidden, out_dim=pool_out,
            )
        else:
            self.transition_pool = SpatialGapPool3d(attention_in, out_dim=pool_out)
        self.motion_proj = None if pooled_dim == d_model else nn.Linear(pooled_dim, d_model)
        self.motion_time = ContinuousTimeEncoding(d_model, max_cycles=time_max_cycles)
        self.motion_transformer = TemporalEncoder(
            d_model, nhead, temporal_layers, dim_feedforward, temporal_dropout,
        )
        # Optional replacement for linear map_motion_to_state. Absent on the default path.
        if motion_align == "cross_deform":
            self.cross_motion_align = CrossTemporalDeformAlign(
                d_model, nhead=nhead, n_points=self.deform_n_points, dropout=temporal_dropout,
            )
        else:
            self.cross_motion_align = None

        self.selector_fusion = nn.Sequential(
            nn.Linear(d_model * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.LayerNorm(fusion_dim),
        )
        self.start_head = IntervalEndpointHead(fusion_dim, selector_hidden, selector_dropout)
        self.end_head = IntervalEndpointHead(fusion_dim, selector_hidden, selector_dropout)

        feature_dim = _feature_dim(variant, d_model, self.head_features)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def _appearance_volumes(self, volumes):
        """Optionally append per-frame raw difference as a second stem channel.

        Channel 0 stays intensity. Channel 1 is ``|v_t - v_{t-1}|`` (default) or
        signed ``v_t - v_{t-1}``. Frame 0 reuses the first pair ``v_1 - v_0`` so
        every valid frame has a motion channel (avoids a BN subpopulation with
        ch1≡0). Sector / raw-motion branches still consume the original
        one-channel volumes.
        """
        if not self.use_stem_frame_diff:
            return volumes
        diff = volumes.new_zeros(volumes.shape)
        if volumes.shape[1] >= 2:
            pair = volumes[:, 1:] - volumes[:, :-1]
            diff[:, 1:] = pair
            diff[:, 0] = pair[:, 0]
        motion = diff.abs() if self.stem_diff_mode == "abs" else diff
        return torch.cat([volumes, motion], dim=2)

    def _encode_valid(self, volumes, frame_mask):
        """Run the 3D encoder on real volumes only and scatter maps back."""
        batch, steps = frame_mask.shape
        flat_mask = frame_mask.reshape(-1)
        index = flat_mask.nonzero(as_tuple=False).flatten()
        if index.numel() == 0:
            raise ValueError("batch has no valid volumes")
        appearance = self._appearance_volumes(volumes)
        valid = appearance.reshape(batch * steps, *appearance.shape[2:]).index_select(0, index)
        local_parts, global_parts = [], []
        for chunk in valid.split(self.encoder_chunk):
            if self.checkpoint_encoder:
                layer2, layer3, layer4 = self.encoder.forward_stages_checkpointed(chunk)
            else:
                layer2, layer3, layer4 = self.encoder.forward_stages(chunk)
            local_parts.append(layer2 if self.local_layer == "layer2" else layer3)
            global_parts.append(layer4)
        local = torch.cat(local_parts, dim=0)
        global_map = torch.cat(global_parts, dim=0)
        if local.shape[1] != self.local_channels or global_map.shape[1] != self.global_channels:
            raise RuntimeError(
                f"backbone channels {(local.shape[1], global_map.shape[1])} "
                f"!= {(self.local_channels, self.global_channels)}"
            )
        return local, global_map, index

    def _pool_global(self, global_valid, volumes_valid, index, batch, steps):
        """Global anatomy pool. Optional sector-weighted mean; default stays GAP."""
        if self.use_sector_mask_global:
            sector = sector_at_resolution(volumes_valid, global_valid.shape[-3:])
            weight = sector / sector.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
            pooled = (global_valid.float() * weight).sum(dim=(-3, -2, -1)).to(dtype=global_valid.dtype)
        else:
            pooled = self.global_pool(global_valid).flatten(1)
        pooled = self.global_proj(pooled)
        return _scatter(pooled, index, batch, steps)

    def _pool_local(self, local_valid, volumes_valid, index, batch, steps, return_map):
        sector = None
        if self.use_sector_mask:
            sector = sector_at_resolution(volumes_valid, local_valid.shape[-3:])
        pooled, attn = self.local_pool(local_valid, sector=sector, return_map=return_map)
        tokens = _scatter(pooled, index, batch, steps)
        maps = _scatter(attn, index, batch, steps) if attn is not None else None
        return tokens, maps

    def _chunked(self, module, tensor):
        parts = []
        for chunk in tensor.split(self.transition_chunk):
            if self.training and self.checkpoint_encoder and torch.is_grad_enabled():
                parts.append(checkpoint(module, chunk, use_reentrant=False))
            else:
                parts.append(module(chunk))
        return torch.cat(parts, dim=0)

    def _encode_raw_motion(self, raw_pairs):
        encoded = self._chunked(self.raw_difference_encoder, raw_pairs)
        return self._chunked(self.raw_motion_proj, encoded)

    def _fuse_motion(self, feature_maps, raw_maps):
        parts = []
        feat_chunks = feature_maps.split(self.transition_chunk)
        raw_chunks = raw_maps.split(self.transition_chunk)
        for feature_chunk, raw_chunk in zip(feat_chunks, raw_chunks):
            fused = torch.cat([feature_chunk, raw_chunk], dim=1)
            if self.training and self.checkpoint_encoder and torch.is_grad_enabled():
                parts.append(checkpoint(self.motion_fusion, fused, use_reentrant=False))
            else:
                parts.append(self.motion_fusion(fused))
        return torch.cat(parts, dim=0)

    def _motion_maps(self, volumes, local_dense, pair_index, batch, transitions):
        feature_maps = raw_maps = None
        if self.use_feature_transition:
            channels = local_dense.shape[2]
            spatial = local_dense.shape[-3:]
            left = local_dense[:, :-1].reshape(batch * transitions, channels, *spatial).index_select(0, pair_index)
            right = local_dense[:, 1:].reshape(batch * transitions, channels, *spatial).index_select(0, pair_index)
            transition_in = build_transition_input(left, right, self.transition_representation)
            feature_maps = self._chunked(self.transition_encoder, transition_in)
        if self.use_raw_difference:
            raw_pairs = valid_raw_motion_pairs(volumes, pair_index, batch, transitions)
            raw_maps = self._encode_raw_motion(raw_pairs)
        if self.use_raw_difference and self.use_feature_transition:
            motion_maps = self._fuse_motion(feature_maps, raw_maps)
        elif self.use_raw_difference:
            motion_maps = self._chunked(self.raw_to_attention, raw_maps)
        else:
            motion_maps = feature_maps
        return raw_maps, feature_maps, motion_maps

    def _transitions(self, volumes, local_dense, frame_mask, return_map):
        batch, steps = frame_mask.shape
        motion_mask = frame_mask[:, :-1] & frame_mask[:, 1:]
        transitions = steps - 1
        viz = {
            "raw_motion_feature": None,
            "feature_transition_feature": None,
            "fused_motion_feature": None,
        }
        if transitions <= 0:
            empty = local_dense.new_zeros(batch, 0, self.d_model)
            return empty, motion_mask, None, viz
        pair_index = motion_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        spatial = local_dense.shape[-3:]
        if pair_index.numel() == 0 or not (self.use_raw_difference or self.use_feature_transition):
            tokens = local_dense.new_zeros(batch, transitions, self.d_model)
            maps = local_dense.new_zeros(batch, transitions, 1, *spatial) if return_map else None
            return tokens, motion_mask, maps, viz
        raw_maps, feature_maps, motion_maps = self._motion_maps(
            volumes, local_dense, pair_index, batch, transitions,
        )
        sector = None
        if self.use_sector_mask:
            left = volumes[:, :-1].reshape(batch * transitions, *volumes.shape[2:]).index_select(0, pair_index)
            right = volumes[:, 1:].reshape(batch * transitions, *volumes.shape[2:]).index_select(0, pair_index)
            spatial = motion_maps.shape[-3:]
            sector = 0.5 * (
                sector_at_resolution(left, spatial) + sector_at_resolution(right, spatial)
            )
        pooled_parts, attn_parts = [], []
        map_chunks = list(motion_maps.split(self.transition_chunk))
        if sector is None:
            sector_chunks = [None] * len(map_chunks)
        else:
            sector_chunks = list(sector.split(self.transition_chunk))
        for chunk, sector_chunk in zip(map_chunks, sector_chunks):
            pooled, attn = self.transition_pool(chunk, sector=sector_chunk, return_map=return_map)
            if self.motion_proj is not None:
                pooled = self.motion_proj(pooled)
            pooled_parts.append(pooled)
            if return_map:
                attn_parts.append(attn)
        tokens = _scatter(torch.cat(pooled_parts, dim=0), pair_index, batch, transitions)
        maps = _scatter(torch.cat(attn_parts, dim=0), pair_index, batch, transitions) if return_map else None
        if return_map:
            if raw_maps is not None:
                viz["raw_motion_feature"] = _scatter(raw_maps, pair_index, batch, transitions)
            if feature_maps is not None:
                viz["feature_transition_feature"] = _scatter(feature_maps, pair_index, batch, transitions)
            viz["fused_motion_feature"] = _scatter(motion_maps, pair_index, batch, transitions)
        return tokens, motion_mask, maps, viz

    def _empty_output(self, batch, device):
        del batch, device
        return {
            "logits": None,
            "H_state": None,
            "H_motion": None,
            "phase_weights": None,
            "phase_weights_raw": None,
            "interval_prob": None,
            "interval_entropy": None,
            "expected_start": None,
            "expected_end": None,
            "expected_phase_ratio": None,
            "z_phase_anatomy": None,
            "z_phase_motion": None,
            "z_global_anatomy": None,
            "local_spatial_attention": None,
            "transition_spatial_attention": None,
            "raw_motion_feature": None,
            "feature_transition_feature": None,
            "fused_motion_feature": None,
        }

    def _logits(self, features):
        return self.classifier(features).squeeze(-1)

    def forward(self, volumes, frame_mask, lengths=None, return_visualization=False):
        """volumes [B, T, 1, 64, 64, 64], frame_mask [B, T] bool, lengths [B]."""
        if volumes.ndim != 6 or volumes.shape[2:] != (1, 64, 64, 64):
            raise ValueError(f"expected [B, T, 1, 64, 64, 64], got {tuple(volumes.shape)}")
        if frame_mask.shape != volumes.shape[:2]:
            raise ValueError("frame_mask must be [B, T]")
        if frame_mask.dtype != torch.bool:
            frame_mask = frame_mask.bool()
        derived = _assert_prefix_mask(frame_mask)
        if lengths is None:
            lengths = derived
        elif not torch.equal(lengths.to(device=derived.device), derived):
            raise ValueError("lengths do not match frame_mask")
        lengths = derived

        batch, steps = frame_mask.shape
        local_valid, global_valid, index = self._encode_valid(volumes, frame_mask)
        volumes_valid = volumes.reshape(batch * steps, *volumes.shape[2:]).index_select(0, index)
        z_global_frame = self._pool_global(global_valid, volumes_valid, index, batch, steps)
        z_global_frame = z_global_frame * frame_mask.unsqueeze(-1).to(dtype=z_global_frame.dtype)
        output = self._empty_output(batch, volumes.device)

        if self.variant == "M0":
            pooled = _masked_mean(z_global_frame, frame_mask)
            output["logits"] = self._logits(pooled)
            output["z_global_anatomy"] = pooled
            return output

        z_local, local_attn = self._pool_local(
            local_valid, volumes_valid, index, batch, steps, return_visualization,
        )
        z_local = z_local * frame_mask.unsqueeze(-1).to(dtype=z_local.dtype)
        state = self.state_token(torch.cat([z_local, z_global_frame], dim=-1))
        tau = normalized_cardiac_time(lengths, steps, volumes.device)
        state = state + self.state_time(tau).to(dtype=state.dtype)
        state = state * frame_mask.unsqueeze(-1).to(dtype=state.dtype)
        if self.temporal_mixer == "dense":
            h_state = self.state_transformer(state, key_padding_mask=~frame_mask)
        else:
            h_state = self.state_deform_encoder(state, key_padding_mask=~frame_mask)
        output["H_state"] = h_state
        output["local_spatial_attention"] = local_attn
        output["temporal_mixer"] = self.temporal_mixer
        output["motion_align"] = self.motion_align

        if self.variant == "M1":
            pooled = _masked_mean(h_state, frame_mask)
            output["logits"] = self._logits(pooled)
            output["z_global_anatomy"] = pooled
            return output

        local_dense = _scatter(local_valid, index, batch, steps)
        motion, motion_mask, transition_attn, motion_viz = self._transitions(
            volumes, local_dense, frame_mask, return_visualization,
        )
        output.update(motion_viz)
        if self.use_raw_difference or self.use_feature_transition:
            tau_motion = 0.5 * (tau[:, :-1] + tau[:, 1:])
            tau_motion = tau_motion * motion_mask.to(dtype=tau_motion.dtype)
            motion = motion + self.motion_time(tau_motion).to(dtype=motion.dtype)
            if motion.shape[1] > 0:
                motion = motion * motion_mask.unsqueeze(-1).to(dtype=motion.dtype)
                h_motion = self.motion_transformer(motion, key_padding_mask=~motion_mask)
            else:
                h_motion = motion
        else:
            h_motion = motion
        output["H_motion"] = h_motion
        output["transition_spatial_attention"] = transition_attn

        if self.variant == "M2":
            z_state = _masked_mean(h_state, frame_mask)
            z_motion = _masked_mean(h_motion, motion_mask)
            output["logits"] = self._logits(torch.cat([z_state, z_motion], dim=-1))
            output["z_global_anatomy"] = z_state
            output["z_phase_motion"] = z_motion
            return output

        if self.motion_align == "linear":
            h_motion_state = map_motion_to_state(h_motion, lengths)
        else:
            h_motion_state = self.cross_motion_align(
                h_state, h_motion, frame_mask, motion_mask,
            )
        fused = self.selector_fusion(torch.cat([h_state, h_motion_state], dim=-1))
        fused = fused * frame_mask.unsqueeze(-1).to(dtype=fused.dtype)
        start_logits = self.start_head(fused)
        end_logits = self.end_head(fused)
        interval_prob, _valid = interval_distribution(start_logits, end_logits, frame_mask)
        phase_weights, phase_raw = phase_membership(interval_prob, frame_mask)
        diagnostics = interval_diagnostics(interval_prob, lengths)
        # Selector still runs for diagnostics; classification may ignore its weights.
        if self.temporal_pool == "phase":
            z_phase_anatomy = (phase_weights.unsqueeze(-1) * h_state.float()).sum(dim=1)
            motion_weights = motion_phase_weights(phase_raw, motion_mask)
            z_phase_motion = (motion_weights.unsqueeze(-1) * h_motion.float()).sum(dim=1)
        else:
            z_phase_anatomy = _masked_mean(h_state.float(), frame_mask)
            z_phase_motion = _masked_mean(h_motion.float(), motion_mask)
        z_global = _masked_mean(h_state.float(), frame_mask)
        features = _assemble_head_features(
            z_phase_anatomy, z_phase_motion, z_global, self.head_features,
        )
        output["logits"] = self._logits(features.to(dtype=next(self.classifier.parameters()).dtype))
        output["phase_weights"] = phase_weights
        output["phase_weights_raw"] = phase_raw
        output["temporal_pool"] = self.temporal_pool
        output["interval_prob"] = interval_prob
        output["interval_entropy"] = diagnostics["interval_entropy"]
        output["expected_start"] = diagnostics["expected_start"]
        output["expected_end"] = diagnostics["expected_end"]
        output["expected_phase_ratio"] = diagnostics["expected_phase_ratio"]
        output["z_phase_anatomy"] = z_phase_anatomy
        output["z_phase_motion"] = z_phase_motion
        output["z_global_anatomy"] = z_global
        return output


def build_model(cfg):
    """Build AnatomyMotion4DNet from the nested experiment config."""
    model_cfg = cfg["model"]
    spatial = cfg["spatial_encoder"]
    local = cfg["local_anatomy"]
    state = cfg["state_transformer"]
    transition = cfg["spatial_transition"]
    selector = cfg["interval_selector"]
    classifier = cfg["classifier"]
    motion = cfg["motion_transformer"]
    fusion = cfg["selector_fusion"]
    transition_attention = cfg["transition_attention"]
    for key in ("d_model", "nhead", "num_layers", "dim_feedforward", "dropout"):
        if motion[key] != state[key]:
            raise ValueError(
                f"motion_transformer.{key} differs from state_transformer; "
                "the two modules do not share weights, and this build keeps the specified width"
            )
    d_model = int(state["d_model"])
    if int(fusion["input_dim"]) != 2 * d_model:
        raise ValueError(
            f"selector_fusion.input_dim is {fusion['input_dim']}, "
            f"but state and motion tokens concatenate to {2 * d_model}"
        )
    hidden_channels = int(transition["hidden_channels"])
    output_channels = int(transition.get("output_channels", hidden_channels))
    temporal_opts = cfg.get("temporal_options", {})
    return AnatomyMotion4DNet(
        variant=model_cfg["variant"],
        local_layer=spatial["local_layer"],
        local_pooling=local["pooling"],
        transition_representation=transition["representation"],
        transition_pooling=cfg["transition_attention"]["pooling"],
        d_model=d_model,
        nhead=state["nhead"],
        temporal_layers=state["num_layers"],
        dim_feedforward=state["dim_feedforward"],
        temporal_dropout=state["dropout"],
        transition_dropout=transition["dropout"],
        selector_dropout=selector["dropout"],
        selector_hidden=selector["hidden_dim"],
        classifier_dropout=classifier["dropout"],
        classifier_hidden=classifier["hidden_dim"],
        attention_hidden=local["attention_hidden"],
        transition_hidden_channels=hidden_channels,
        transition_output_channels=output_channels,
        transition_attention_hidden=int(transition_attention["attention_hidden"]),
        transition_attention_dim=int(transition_attention["output_dim"]),
        fusion_dropout=float(fusion["dropout"]),
        fusion_dim=int(fusion["output_dim"]),
        use_raw_difference=bool(cfg.get("motion_branch", {}).get("use_raw_difference", True)),
        use_feature_transition=bool(cfg.get("motion_branch", {}).get("use_feature_transition", True)),
        head_features=classifier.get("head_features"),
        temporal_pool=str(classifier.get("temporal_pool", "phase")),
        use_sector_mask=bool(cfg.get("spatial_prior", {}).get("use_sector_mask", False)),
        use_sector_mask_global=bool(cfg.get("spatial_prior", {}).get("use_sector_mask_global", False)),
        use_stem_frame_diff=bool(cfg.get("appearance_motion", {}).get("use_stem_frame_diff", False)),
        stem_diff_mode=str(cfg.get("appearance_motion", {}).get("stem_diff_mode", "abs")),
        temporal_mixer=str(temporal_opts.get("temporal_mixer", "dense")),
        motion_align=str(temporal_opts.get("motion_align", "linear")),
        deform_n_points=int(temporal_opts.get("deform_n_points", 4)),
        checkpoint_encoder=spatial.get("checkpoint_encoder", True),
        encoder_chunk=spatial.get("encoder_chunk", 8),
        transition_chunk=transition.get("chunk", 8),
        time_max_cycles=cfg.get("time_encoding", {}).get("max_cycles", 32.0),
    )
