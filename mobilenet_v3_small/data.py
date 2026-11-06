"""ResNet-compatible data loading; real data must stay on Perlmutter."""
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.data import load_split  # noqa: E402


def load_data(data_root, fold=0):
    # The team's shared loader: same split and checks for every model.
    return load_split(fold, data_root)


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
