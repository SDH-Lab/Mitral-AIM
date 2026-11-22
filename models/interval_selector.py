"""One continuous diagnostic interval, weakly supervised by the clip label."""
import torch
from torch import nn


EPS = 1e-6


class IntervalEndpointHead(nn.Module):
    def __init__(self, dim=256, hidden_dim=128, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden):
        return self.net(hidden).squeeze(-1)


def interval_distribution(start_logits, end_logits, frame_mask):
    """Softmax over valid pairs ``i <= j``. Invalid pairs are exactly zero.

    Computed in fp32. ``frame_mask`` is True on real frames.
    """
    start = start_logits.float()
    end = end_logits.float()
    scores = start.unsqueeze(2) + end.unsqueeze(1)
    steps = frame_mask.shape[1]
    index = torch.arange(steps, device=frame_mask.device)
    order = index.view(1, steps, 1) <= index.view(1, 1, steps)
    valid = frame_mask.unsqueeze(2) & frame_mask.unsqueeze(1) & order
    scores = scores.masked_fill(~valid, float("-inf"))
    flat = torch.softmax(scores.view(scores.shape[0], -1), dim=-1)
    prob = flat.view_as(scores)
    prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = prob * valid.to(dtype=prob.dtype)
    return prob, valid


def phase_membership(interval_prob, frame_mask):
    """``w_t = sum_{i <= t <= j} P(i, j)``, then normalize over valid frames."""
    cumulative = torch.cumsum(interval_prob, dim=1)
    suffix = torch.cumsum(cumulative.flip(-1), dim=-1).flip(-1)
    raw = suffix.diagonal(dim1=-2, dim2=-1).contiguous()
    raw = raw * frame_mask.to(dtype=raw.dtype)
    weights = raw / (raw.sum(dim=-1, keepdim=True) + EPS)
    weights = weights * frame_mask.to(dtype=weights.dtype)
    return weights, raw


def interval_diagnostics(interval_prob, lengths):
    """Expected start, end, phase ratio, and normalized interval entropy.

    Start and end are normalized by ``T - 1`` into ``[0, 1]``. The phase ratio
    uses the valid length ``T``, not the padded length.
    """
    prob = interval_prob.float()
    batch, steps, _ = prob.shape
    device = prob.device
    index = torch.arange(steps, device=device, dtype=prob.dtype)
    lengths_f = lengths.to(device=device, dtype=prob.dtype).clamp_min(1)
    denom = (lengths_f - 1).clamp_min(1)
    normalized = index.view(1, steps) / denom.view(batch, 1)
    marginal_start = prob.sum(dim=-1)
    marginal_end = prob.sum(dim=-2)
    expected_start = (marginal_start * normalized).sum(dim=-1)
    expected_end = (marginal_end * normalized).sum(dim=-1)

    span = index.view(1, 1, steps) - index.view(1, steps, 1) + 1
    span = span.clamp_min(0) / lengths_f.view(batch, 1, 1)
    expected_phase_ratio = (prob * span).sum(dim=(-2, -1))

    entropy = -(prob * (prob + EPS).log()).sum(dim=(-2, -1))
    n_intervals = lengths_f * (lengths_f + 1) / 2
    interval_entropy = entropy / n_intervals.log().clamp_min(EPS)
    return {
        "expected_start": expected_start,
        "expected_end": expected_end,
        "expected_phase_ratio": expected_phase_ratio,
        "interval_entropy": interval_entropy,
    }


def motion_phase_weights(phase_raw, motion_mask):
    """Average adjacent phase mass, drop invalid transitions, then normalize."""
    pair = 0.5 * (phase_raw[:, :-1] + phase_raw[:, 1:])
    pair = pair * motion_mask.to(dtype=pair.dtype)
    return pair / (pair.sum(dim=-1, keepdim=True) + EPS)
