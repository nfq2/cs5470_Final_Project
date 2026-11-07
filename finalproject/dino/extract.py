"""Run the frozen backbone once over every image and save its features.

python extract.py --model vit_small_patch14_dinov2.lvd142m --size 448
python extract.py --size 448 --tiles 2    # each image cut into 2x2 tiles, each encoded at 448

Tiles keep the wire at close to native resolution: a 1065-pixel frame resized to 448 turns
a 2-3 px wire into 1 px, and each 14 px DINO patch then covers 33 px of the original; with
2x2 tiles a patch covers 16 px. Patch tokens of the tiles are concatenated per image, so
the heads see [416, 7, tiles^2 * P, D] and need no change.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

# Use Saesha's loader so every model reads the same dataset copy and split. Appended,
# not prepended, so this folder's train.py wins over resnet/train.py.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.data import data_root  # noqa: E402

DATA_DIR = data_root()

FEATURE_ROOT = Path(os.environ["PSCRATCH"]) / "finalproject/dino/features"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def variant_name(model_name, size, tiles=1):
    # vit_small_patch14_dinov2.lvd142m at 448 -> vit_small_dinov2_448 (or ..._448_tiles2)
    name = model_name.split(".")[0].replace("_patch14", "") + f"_{size}"
    return name if tiles == 1 else f"{name}_tiles{tiles}"


def build_backbone(model_name, size, device):
    # img_size resamples DINOv2's position embeddings to the new patch grid.
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=size)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone.eval().to(device)


def read_images(case):
    """The seven task images (goal, f1..f6) as uint8 [7, H, W, 3]."""
    images = []
    for path in case["paths"]:
        with Image.open(DATA_DIR / path) as image:
            images.append(np.asarray(image.convert("RGB")))
    return torch.from_numpy(np.stack(images))


def tile(images, tiles):
    """uint8 [N, H, W, 3] -> [N * tiles^2, H/tiles, W/tiles, 3], tiles in row-major order."""
    if tiles == 1:
        return images
    n, h, w, c = images.shape
    h, w = h // tiles * tiles, w // tiles * tiles  # drop at most tiles-1 edge pixels
    x = images[:, :h, :w].reshape(n, tiles, h // tiles, tiles, w // tiles, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(n * tiles * tiles, h // tiles, w // tiles, c)


def preprocess(images, size, tiles=1):
    """uint8 [N, H, W, 3] on the GPU -> normalized float [N * tiles^2, 3, size, size]."""
    x = tile(images, tiles).permute(0, 3, 1, 2).float().div(255)
    x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def encode(backbone, pixels):
    """Patch tokens [N, P, D] and CLS token [N, D]."""
    tokens = backbone.forward_features(pixels)
    return tokens[:, backbone.num_prefix_tokens:], tokens[:, 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--tiles", type=int, default=1, help="cut each image into tiles x tiles")
    args = parser.parse_args()
    assert args.size % 14 == 0, "DINOv2 uses 14-pixel patches; choose a multiple of 14."
    assert torch.cuda.is_available(), "Allocate a GPU before running extract.py."
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with open(DATA_DIR / "cases.jsonl") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    backbone = build_backbone(args.model, args.size, device)

    patches = cls = None
    start = time.perf_counter()
    with torch.no_grad():
        for i, case in enumerate(cases):
            images = read_images(case).to(device)
            patch_tokens, cls_tokens = encode(backbone, preprocess(images, args.size, args.tiles))
            # Tiles of one image become one long patch sequence; CLS is averaged over tiles.
            patch_tokens = patch_tokens.reshape(images.shape[0], -1, patch_tokens.shape[-1])
            cls_tokens = cls_tokens.reshape(images.shape[0], -1, cls_tokens.shape[-1]).mean(1)
            if patches is None:
                patches = torch.empty((len(cases), 7, *patch_tokens.shape[1:]), dtype=torch.float16)
                cls = torch.empty((len(cases), 7, cls_tokens.shape[-1]), dtype=torch.float16)
            patches[i] = patch_tokens.half().cpu()
            cls[i] = cls_tokens.half().cpu()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(cases)} cases", flush=True)

    out = FEATURE_ROOT / variant_name(args.model, args.size, args.tiles)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"patches": patches, "cls": cls}, out / "features.pt")
    meta = {"model": args.model, "size": args.size, "tiles": args.tiles, "shape": list(patches.shape),
            "precision": "FP32 compute, TF32 disabled, saved as FP16",
            "case_ids": [case["case_id"] for case in cases]}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f)
    print(f"Saved {tuple(patches.shape)} to {out} in {time.perf_counter() - start:.0f} s")
