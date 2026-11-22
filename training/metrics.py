"""Thresholded classification metrics. The threshold comes from validation only."""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def _binary_counts(labels, probabilities, threshold):
    y = np.asarray(labels).astype(int)
    pred = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp


def f1_score(labels, probabilities, threshold):
    tn, fp, fn, tp = _binary_counts(labels, probabilities, threshold)
    del tn
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def select_threshold(labels, probabilities):
    """Highest validation F1. Ties keep the higher threshold."""
    probs = np.asarray(probabilities, dtype=np.float64)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.unique(probs):
        score = f1_score(labels, probs, float(threshold))
        if score > best_f1 or (score == best_f1 and float(threshold) > best_threshold):
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics(labels, probabilities, threshold):
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise ValueError("AUROC requires both classes")
    tn, fp, fn, tp = _binary_counts(y, p, threshold)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "accuracy": float(np.mean((p >= threshold) == y)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
