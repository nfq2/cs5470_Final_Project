"""One loader for every model, so all branches train and evaluate on the same cases.

    from common.data import load_split, ShortHorizonDataset

    cases, split = load_split(fold=0)                       # split["train"] etc. index cases
    train = ShortHorizonDataset(cases, split["train"], transform=my_transform)

Real data stays on Perlmutter. The split is the one supplied in splits.json; nothing here
re-splits. `fold` is the stored fold ID, not the position of the entry in the file.
"""
import json
import os
from pathlib import Path

GROUPS = ("train", "validation", "test")
ROLES = ["goal_map", "history_1", "history_2", "history_3", "history_4", "history_5", "now"]


def data_root(root=None):
    """Dataset directory: the argument, else $SHORTHORIZON_ROOT, else the team copy."""
    if root is None:
        root = os.environ.get("SHORTHORIZON_ROOT") or Path(os.environ["PSCRATCH"]) / "finalproject/shorthorizon2"
    return Path(root).expanduser().resolve()


def load_cases(root=None):
    with open(data_root(root) / "cases.jsonl") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    for case in cases:
        if len(case["paths"]) != 7 or case["label"] not in (0, 1):
            raise ValueError("Each case needs seven images and a RETRACT=0 / ADVANCE=1 label")
        if len(case["history_offsets_s"]) != 6 or len(case["depth_relative_mm"]) != 6:
            raise ValueError("Each case needs six time offsets and six insertion depths")
    return cases


def fold_ids(root=None):
    with open(data_root(root) / "splits.json") as f:
        return sorted(item["fold"] for item in json.load(f))


def load_split(fold=0, root=None):
    """(cases, split) for one fold. Fails if groups overlap or share an episode."""
    cases = load_cases(root)
    with open(data_root(root) / "splits.json") as f:
        matches = [item for item in json.load(f) if item["fold"] == fold]
    if len(matches) != 1:
        raise ValueError(f"fold {fold} must identify exactly one supplied split")
    split = matches[0]
    indices = [i for group in GROUPS for i in split[group]]
    if any(type(i) is not int or not 0 <= i < len(cases) for i in indices):
        raise ValueError("Split contains an invalid case index")
    if any(not split[group] for group in GROUPS) or len(indices) != len(set(indices)):
        raise ValueError("Split groups must be nonempty and disjoint")
    # A procedure must not appear in more than one group.
    episodes = [{cases[i]["episode"] for i in split[group]} for group in GROUPS]
    if episodes[0] & episodes[1] or episodes[0] & episodes[2] or episodes[1] & episodes[2]:
        raise ValueError("Overlapping episodes in split")
    return cases, split


def load_all_folds(root=None):
    """Every fold's split, for pooled evaluation. Fails unless the test groups together
    cover each case exactly once, which is what makes pooled test numbers meaningful."""
    cases = load_cases(root)
    splits = [load_split(fold, root)[1] for fold in fold_ids(root)]
    tested = sorted(i for split in splits for i in split["test"])
    if tested != list(range(len(cases))):
        raise ValueError("Test groups do not cover every case exactly once")
    return cases, splits


class ShortHorizonDataset:
    """One item: seven images (goal first, then f1..f6), timing, depth, label.

    transform takes a PIL RGB image and returns a CHW tensor; it must be deterministic or
    synchronized across the seven images. clean=True reads the goal map without the target
    circle drawn on it (clean_paths) instead of the task images (paths).
    """

    def __init__(self, cases, indices, transform, root=None, clean=False):
        self.cases = [cases[i] for i in indices]
        self.transform = transform
        self.root = data_root(root)
        self.key = "clean_paths" if clean else "paths"

    def __len__(self):
        return len(self.cases)

    def _image(self, path):
        from PIL import Image

        with Image.open(self.root / path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index):
        import torch

        case = self.cases[index]
        if case.get("image_roles", ROLES) != ROLES:
            raise ValueError("Expected goal map followed by six chronological frames")
        images = [self._image(path) for path in case[self.key]]
        return {"images": torch.stack(images),
                "history_offsets_s": torch.tensor(case["history_offsets_s"], dtype=torch.float32),
                "depth_relative_mm": torch.tensor(case["depth_relative_mm"], dtype=torch.float32),
                "label": torch.tensor(case["label"], dtype=torch.long)}
