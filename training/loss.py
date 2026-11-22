"""Classification loss plus optional inactive phase regularizers."""
import torch
from torch import nn


class AnatomyMotionLoss(nn.Module):
    """Per-sample BCE, with optional entropy and phase-length penalties.

    Both penalties are zero unless their lambdas are set. They stay zero
    until ``aux_start_epoch`` so a later ablation can warm them up.
    """

    def __init__(
        self,
        pos_weight=None,
        lambda_interval_entropy=0.0,
        lambda_phase_length=0.0,
        min_phase_ratio=0.05,
        max_phase_ratio=0.80,
        aux_start_epoch=10,
    ):
        super().__init__()
        self.lambda_interval_entropy = float(lambda_interval_entropy)
        self.lambda_phase_length = float(lambda_phase_length)
        self.min_phase_ratio = float(min_phase_ratio)
        self.max_phase_ratio = float(max_phase_ratio)
        self.aux_start_epoch = int(aux_start_epoch)
        weight = None if pos_weight is None else torch.tensor([float(pos_weight)])
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=weight)

    def forward(self, outputs, labels, epoch=0):
        logits = outputs["logits"].float().view(-1)
        targets = labels.float().view(-1)
        per_sample = self.bce(logits, targets)
        extra = torch.zeros_like(per_sample)
        if epoch < self.aux_start_epoch:
            return per_sample, extra
        if self.lambda_interval_entropy:
            entropy = outputs["interval_entropy"]
            if entropy is None:
                raise RuntimeError("interval-entropy loss requires the phase selector (M3 or M4)")
            extra = extra + self.lambda_interval_entropy * entropy.float().view(-1)
        if self.lambda_phase_length:
            ratio = outputs["expected_phase_ratio"]
            if ratio is None:
                raise RuntimeError("phase-length loss requires the phase selector (M3 or M4)")
            penalty = torch.relu(self.min_phase_ratio - ratio.float()) + torch.relu(ratio.float() - self.max_phase_ratio)
            extra = extra + self.lambda_phase_length * penalty.view(-1)
        return per_sample, extra
