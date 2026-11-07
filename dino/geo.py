"""Vessel graph from each goal map, and distance to the target along the vessels.

python geo.py            # all 416 cases, 8 processes, CPU only; prints summary statistics
python geo.py --limit 20

Per case it saves `geo/<case_id>.npz` in $PSCRATCH with the vessel skeleton and a map of
geodesic (along-vessel) distance in pixels from every skeleton pixel to the target. The
target is fixed per episode, but each case has its own goal map, so this runs per case.
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, frangi
from skimage.graph import MCP_Geometric
from skimage.morphology import binary_closing, disk, remove_small_objects, skeletonize

DATA_DIR = Path(os.environ["PSCRATCH"]) / "finalproject/shorthorizon2"
GEO_DIR = Path(os.environ["PSCRATCH"]) / "finalproject/dino/geo"
VESSEL_SIGMAS = np.arange(3, 14, 2)  # contrast-filled vessels are ~10-40 px wide at full res
OFF_VESSEL_COST = 25.0  # per pixel off the vessel mask; 1.0 on it. Gaps cost, but stay finite.


def load_gray(relative_path):
    with Image.open(DATA_DIR / relative_path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255


def ui_mask(shape):
    """False where the simulator draws its overlays (eye icon and markers, top right)."""
    h, w = shape
    mask = np.ones(shape, bool)
    mask[: int(0.30 * h), int(0.86 * w):] = False
    return mask


def vessel_skeleton(gray):
    """Boolean skeleton of the contrast-filled vessels, plus the thresholded mask."""
    response = frangi(gray, sigmas=VESSEL_SIGMAS, black_ridges=True)
    response[~ui_mask(gray.shape)] = 0
    high, low = np.percentile(response, [97, 92])
    mask = apply_hysteresis_threshold(response, low, high)
    mask = binary_closing(mask, disk(3))
    mask = remove_small_objects(mask, 800)
    return skeletonize(mask), mask


def snap(mask, x, y):
    """Nearest True pixel of mask to (x, y): returns (y, x, distance_px)."""
    _, indices = ndi.distance_transform_edt(~mask, return_indices=True)
    sy, sx = indices[:, y, x]
    return int(sy), int(sx), float(np.hypot(sy - y, sx - x))


def geodesic(mask, start_yx):
    """Minimum travel cost from start to every pixel: 1 per pixel along vessels,
    OFF_VESSEL_COST per pixel elsewhere. Finite everywhere, so a tip on a vessel piece the
    detector broke off, or slightly off the mask, still gets a sensible distance."""
    cost = np.where(mask, 1.0, OFF_VESSEL_COST).astype(np.float64)
    dist, _ = MCP_Geometric(cost, fully_connected=True).find_costs([start_yx])
    return dist.astype(np.float32)


def process(case):
    gray = load_gray(case["clean_paths"][0])  # goal_raw.png: no target circle drawn on it
    skeleton, mask = vessel_skeleton(gray)
    x, y = (int(round(v)) for v in case["goal_xy"])
    labels, n_components = ndi.label(skeleton, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    sy, sx, snap_px = snap(mask, x, y) if mask.any() else (y, x, np.inf)
    dist = geodesic(mask, (sy, sx))
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GEO_DIR / f"{case['case_id']}.npz", dist=np.minimum(dist, 60000).astype(np.float16),
                        skeleton=np.packbits(skeleton), mask=np.packbits(mask), shape=np.array(mask.shape))
    on_vessel = dist[mask]
    return {"case_id": case["case_id"], "snap_px": snap_px, "skeleton_px": int(skeleton.sum()),
            "mask_frac": float(mask.mean()), "components": int(n_components),
            "largest_frac": float(sizes.max() / sizes.sum()) if len(sizes) else 0.0,
            "vessel_dist_median": float(np.median(on_vessel)), "vessel_dist_max": float(on_vessel.max())}


def percentiles(values, name):
    v = np.asarray(values, dtype=np.float64)
    print(f"{name:<26} min/25/50/75/max: {np.percentile(v, [0, 25, 50, 75, 100]).round(2).tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with open(DATA_DIR / "cases.jsonl") as f:
        cases = [json.loads(line) for line in f if line.strip()][: args.limit]
    with Pool(args.workers) as pool:
        stats = pool.map(process, cases, chunksize=4)
    with open(GEO_DIR / "stats.jsonl", "w") as f:
        for s in stats:
            f.write(json.dumps(s) + "\n")
    print(f"cases: {len(stats)}")
    percentiles([s["mask_frac"] * 100 for s in stats], "vessel mask % of image")
    percentiles([s["skeleton_px"] for s in stats], "skeleton pixels")
    percentiles([s["components"] for s in stats], "components")
    percentiles([s["largest_frac"] for s in stats], "largest component frac")
    percentiles([s["snap_px"] for s in stats], "target->vessel px")
    print("target within 15 px of a vessel:", sum(s["snap_px"] <= 15 for s in stats))
    percentiles([s["vessel_dist_median"] for s in stats], "median dist on vessels")
    percentiles([s["vessel_dist_max"] for s in stats], "max dist on vessels")
