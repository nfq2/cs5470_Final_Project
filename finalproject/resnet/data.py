"""Read the provided examples and episode-based split."""
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.data import load_split  # noqa: E402

DATA_DIR = Path(os.environ["PSCRATCH"]) / "finalproject/shorthorizon2"


def load_data(fold=0):
    # The team's shared loader: same split and checks for every model.
    return load_split(fold, DATA_DIR)


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
