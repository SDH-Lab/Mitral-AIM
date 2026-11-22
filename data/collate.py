"""Pad variable-length clips on the right. Padding never enters the model as valid."""
import torch


def collate_clips(items):
    lengths = [int(item["length"]) for item in items]
    steps = max(lengths)
    batch = len(items)
    volumes = torch.zeros(batch, steps, 1, 64, 64, 64, dtype=torch.float32)
    frame_mask = torch.zeros(batch, steps, dtype=torch.bool)
    for index, item in enumerate(items):
        length = lengths[index]
        volumes[index, :length] = item["volume"]
        frame_mask[index, :length] = True
    return {
        "volumes": volumes,
        "frame_mask": frame_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.float32),
        "patient_id": [item["patient_id"] for item in items],
        "file_name": [item["file_name"] for item in items],
    }
