"""Short-horizon data loading for RepViT. Real data stays on Perlmutter."""

import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.data import load_split, ShortHorizonDataset as SharedDataset  # noqa: E402


class ImagePreprocess:
    """Deterministic RGB resize and normalization, configurable per backbone.

    Defaults are ImageNet normalization and a provisional 224x224 resize.
    Inspect thin-guidewire visibility on Perlmutter before choosing resolution.
    No random crops/flips: independent spatial augmentation would misalign frames.
    """

    def __init__(
        self,
        size: tuple[int, int] = (224, 224),  # height, width
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        if len(size) != 2 or any(n <= 0 for n in size):
            raise ValueError("size must be positive (height, width)")
        if len(mean) != 3 or len(std) != 3 or any(s <= 0 for s in std):
            raise ValueError("mean/std must have three channels; std must be positive")
        self.size = size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(
            (self.size[1], self.size[0]), Image.Resampling.BILINEAR
        )
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(self.size[0], self.size[1], 3)
        pixels = pixels.permute(2, 0, 1).float().div(255)
        return (pixels - self.mean) / self.std


class ShortHorizonDataset(SharedDataset, Dataset):
    """One item = separate goal, six ordered frames, timing/depth, one label.

    Cases and split come from the team's shared loader (common/data.py): split entries
    index cases.jsonl; fold selects the stored fold ID, not the position of the fold in
    splits.json. Identifiers are not model inputs.
    transform must return a CHW tensor and should be deterministic. For spatial
    augmentation, implement a case-level transform that synchronizes all images.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        fold: int = 0,
        transform: Optional[Callable] = None,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.root = Path(root).expanduser().resolve()
        cases, splits = load_split(fold, self.root)
        super().__init__(cases, splits[split],
                         transform if transform is not None else ImagePreprocess(),
                         root=self.root)

    def __len__(self):
        return len(self.cases)

    def _image(self, relative_path):
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Image path must remain within the dataset directory")
        with Image.open(path) as image:
            result = self.transform(image.convert("RGB"))
        if not isinstance(result, torch.Tensor) or result.ndim != 3:
            raise ValueError("Image transform must return a CHW tensor")
        return result

    def __getitem__(self, index):
        case = self.cases[index]
        roles = ["goal_map", "history_1", "history_2", "history_3",
                 "history_4", "history_5", "now"]
        if len(case["paths"]) != 7 or case["image_roles"] != roles:
            raise ValueError("Expected goal map followed by six chronological frames")
        times = torch.tensor(case["history_offsets_s"], dtype=torch.float32)
        depths = torch.tensor(case["depth_relative_mm"], dtype=torch.float32)
        if times.shape != (6,) or depths.shape != (6,):
            raise ValueError("Expected six time offsets and six insertion depths")
        if not torch.isfinite(times).all() or not torch.isfinite(depths).all():
            raise ValueError("Time/depth inputs must be finite")
        if not torch.all(times[1:] >= times[:-1]) or abs(times[-1].item()) > 1e-5:
            raise ValueError("History must be chronological and end at time zero")
        if abs(depths[-1].item()) > 1e-5:
            raise ValueError("Relative depth must end at zero")
        if case["label"] not in (0, 1):
            raise ValueError("Label must be RETRACT=0 or ADVANCE=1")
        batch = super().__getitem__(index)
        images = batch.pop("images")
        return {"goal": images[0], "frames": images[1:], **batch}



def make_dataloader(
    root: str | Path,
    split: str = "train",
    fold: int = 0,
    batch_size: int = 8,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    seed: int = 42,
    pin_memory: bool = False,
) -> DataLoader:
    """Shuffle training only; retain every example, including partial batches."""
    dataset = ShortHorizonDataset(root, split=split, fold=fold, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect batch shapes on Perlmutter")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    loader = make_dataloader(
        args.data_root, split=args.split, fold=args.fold, batch_size=args.batch_size,
        transform=ImagePreprocess(size=(args.image_size, args.image_size)),
    )
    print("Cases:", len(loader.dataset))
    for key, value in next(iter(loader)).items():
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
