"""Phase-weight plots on normalized cardiac time."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_phase_plot(path, phase_weights, expected_start, expected_end, title=""):
    """Plot diagnostic phase weight against normalized cardiac time."""
    weights = np.asarray(phase_weights, dtype=np.float64)
    length = len(weights)
    if length == 0:
        raise ValueError("phase_weights is empty")
    time = np.linspace(0.0, 1.0, length) if length > 1 else np.zeros(1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 3.2))
    axis.plot(time, weights, color="#0f6c8c", linewidth=1.8)
    axis.axvline(float(expected_start), color="#c45c26", linestyle="--", linewidth=1.2, label="expected start")
    axis.axvline(float(expected_end), color="#2f6b3a", linestyle="--", linewidth=1.2, label="expected end")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("normalized cardiac time")
    axis.set_ylabel("diagnostic phase weight")
    axis.set_title(title)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
