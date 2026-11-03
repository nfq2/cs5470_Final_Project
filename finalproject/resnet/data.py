"""Read the provided examples and episode-based split."""
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = Path(os.environ["PSCRATCH"]) / "finalproject/shorthorizon2"


def load_data(fold=0):
    with open(DATA_DIR / "cases.jsonl") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    with open(DATA_DIR / "splits.json") as f:
        split = json.load(f)[fold]

    # A procedure must not appear in more than one group.
    episodes = [{cases[i]["episode"] for i in split[group]}
                for group in ["train", "validation", "test"]]
    assert not (episodes[0] & episodes[1] or episodes[0] & episodes[2]
                or episodes[1] & episodes[2]), "Overlapping episodes in split!"
    return cases, split


def get_history(case):
    # Six depth values followed by their six time offsets.
    values = case["depth_relative_mm"] + case["history_offsets_s"]
    return torch.tensor(values, dtype=torch.float32)


class WireDataset(Dataset):
    def __init__(self, cases, indices, image_size):
        self.cases = cases
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        case = self.cases[self.indices[index]]
        images = []
        # paths is already ordered: goal, f1, f2, f3, f4, f5, f6.
        for path in case["paths"]:
            with Image.open(DATA_DIR / path) as image:
                images.append(self.transform(image.convert("RGB")))
        return torch.stack(images), get_history(case), case["label"]
