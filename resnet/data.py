"""ResNet inputs built on the team's shared dataset and split loader."""
import argparse
import sys
from pathlib import Path

import torch
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.data import ShortHorizonDataset, load_split  # noqa: E402


def load_data(data_root, fold=0):
    return load_split(fold, data_root)


def get_history(case):
    values = case["depth_relative_mm"] + case["history_offsets_s"]
    history = torch.tensor(values, dtype=torch.float32)
    if history.shape != (12,) or not torch.isfinite(history).all():
        raise ValueError("Expected 12 finite history values")
    return history


class WireDataset(ShortHorizonDataset):
    def __init__(self, data_root, cases, indices, image_size):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        super().__init__(cases, indices, transform, root=data_root)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        history = torch.cat([item["depth_relative_mm"], item["history_offsets_s"]])
        return item["images"], history, item["label"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check ResNet input shapes")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    cases, split = load_data(args.data_root, args.fold)
    print({name: len(split[name]) for name in ("train", "validation", "test")})
    images, history, _ = WireDataset(args.data_root, cases, split["train"], 448)[0]
    print("One example: images", tuple(images.shape), "history", tuple(history.shape))
