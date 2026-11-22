"""Compare a discovered interval with a later manual systolic annotation.

Manual phase labels must never be passed to the training loss.
"""


def normalized_to_inclusive_frames(start, end, length):
    """Map normalized cardiac times to inclusive frame indices."""
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return 0, 0
    start_index = int(round(float(start) * (length - 1)))
    end_index = int(round(float(end) * (length - 1)))
    start_index = min(max(start_index, 0), length - 1)
    end_index = min(max(end_index, 0), length - 1)
    if end_index < start_index:
        start_index, end_index = end_index, start_index
    return start_index, end_index


def systolic_phase_mass(phase_weights, systole_mask):
    """Fraction of phase mass that lands inside the systolic frames."""
    weights = [float(value) for value in phase_weights]
    mask = [bool(value) for value in systole_mask]
    if len(weights) != len(mask):
        raise ValueError("phase_weights and systole_mask must have the same length")
    total = sum(weights)
    if total <= 0:
        return 0.0
    inside = sum(weight for weight, flag in zip(weights, mask) if flag)
    return inside / total


def temporal_iou(pred_start, pred_end, ref_start, ref_end):
    """Inclusive-frame IoU of the predicted interval and the systolic interval."""
    predicted = set(range(int(pred_start), int(pred_end) + 1))
    reference = set(range(int(ref_start), int(ref_end) + 1))
    union = predicted | reference
    if not union:
        return 0.0
    return len(predicted & reference) / len(union)


def start_end_error(pred_start, pred_end, ref_start, ref_end):
    """Absolute error of the endpoints in normalized cardiac time."""
    return abs(float(pred_start) - float(ref_start)), abs(float(pred_end) - float(ref_end))
