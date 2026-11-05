"""ResNet-compatible data loading; real data must stay on Perlmutter."""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_data(data_root, fold=0):
    root = Path(data_root).expanduser().resolve()
    with (root / "cases.jsonl").open() as f:
        cases = [json.loads(line) for line in f if line.strip()]
    with (root / "splits.json").open() as f:
        matches = [item for item in json.load(f) if item["fold"] == fold]
    if len(matches) != 1:
        raise ValueError("Choose exactly one supplied fold ID")
    split = matches[0]
    groups = [split[name] for name in ("train", "validation", "test")]
    indices = [i for group in groups for i in group]
    if any(type(i) is not int or not 0 <= i < len(cases) for i in indices):
        raise ValueError("Invalid case index in supplied split")
    if any(not group for group in groups) or len(indices) != len(set(indices)):
        raise ValueError("Split groups must be nonempty and disjoint")
    episodes = [{cases[i]["episode"] for i in group} for group in groups]
    if episodes[0] & episodes[1] or episodes[0] & episodes[2] or episodes[1] & episodes[2]:
        raise ValueError("Overlapping episodes in supplied split")
    return cases, split


def get_history(case):
    depths, times = case["depth_relative_mm"], case["history_offsets_s"]
    if len(depths) != 6 or len(times) != 6:
        raise ValueError("Expected six depths and six time offsets")
    values = torch.tensor(depths + times, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("History values must be finite")
    return values


class WireDataset(Dataset):
    def __init__(self, data_root, cases, indices, image_size):
        self.root = Path(data_root).expanduser().resolve()
        self.cases, self.indices = cases, indices
        # Match finalproject/resnet/data.py: full-image resize, no cropping.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        case = self.cases[self.indices[index]]
        if len(case["paths"]) != 7 or case["label"] not in (0, 1):
            raise ValueError("Expected seven images and a binary label")
        images = []
        for relative in case["paths"]:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Image must be inside the dataset directory")
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))
        return torch.stack(images), get_history(case), case["label"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check shapes without printing sensitive case data")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    cases, split = load_data(args.data_root, args.fold)
    print({name: len(split[name]) for name in ("train", "validation", "test")})
    dataset = WireDataset(args.data_root, cases, split["train"], 448)
    images, history, _ = dataset[0]
    print("One example: images", tuple(images.shape), "history", tuple(history.shape))
