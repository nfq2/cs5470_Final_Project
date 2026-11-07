"""DINOv2 baseline inputs on the team's shared dataset and split loader (same as resnet/)."""
import sys
from pathlib import Path

import torch
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.data import ShortHorizonDataset, load_split  # noqa: E402
from common.protocol import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def load_data(data_root, fold=0):
    return load_split(fold, data_root)


def get_history(case):
    values = case["depth_relative_mm"] + case["history_offsets_s"]  # depth_then_time
    history = torch.tensor(values, dtype=torch.float32)
    if history.shape != (12,) or not torch.isfinite(history).all():
        raise ValueError("Expected 12 finite history values")
    return history


class WireDataset(ShortHorizonDataset):
    def __init__(self, data_root, cases, indices, image_size):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        super().__init__(cases, indices, transform, root=data_root)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        history = torch.cat([item["depth_relative_mm"], item["history_offsets_s"]])
        return item["images"], history, item["label"]
