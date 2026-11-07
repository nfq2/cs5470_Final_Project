"""Label-free wire-tip candidates in f1 and f6.

python tip.py            # all cases; writes geo/tips_auto.json (stays on Perlmutter)

Two candidates per frame, so the click check can tell which idea works:
  A "moving thin line nearest the target": thin-dark-ridge response times frame-to-frame
    change, thresholded; the candidate is the pixel with the smallest along-vessel
    distance to the target. Biased toward "the wire is on the path", so a wrong-branch
    tip may be missed.
  B "wire endpoint": skeletonize the thin-ridge mask, take its endpoints away from the
    image border (the catheter enters at the border), pick the one with the most motion
    around it. No bias toward the target, but confused by other thin structures.
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, frangi, gaussian
from skimage.morphology import remove_small_objects, skeletonize

from geo import DATA_DIR, GEO_DIR, load_gray, ui_mask

WIRE_SIGMAS = [1.0, 1.5, 2.0, 2.5]  # the guidewire is 2-3 px wide at full res
BORDER = 20
FRAMES = {"f1": (1, 2), "f6": (6, 5)}  # frame index in clean_paths, and its motion reference


def wire_response(gray):
    response = frangi(gray, sigmas=WIRE_SIGMAS, black_ridges=True)
    response[~ui_mask(gray.shape)] = 0
    return response


def load_geo(case_id):
    """Along-vessel distance map (float32) and vessel mask (bool)."""
    data = np.load(GEO_DIR / f"{case_id}.npz")
    shape = tuple(data["shape"])
    mask = np.unpackbits(data["mask"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    return data["dist"].astype(np.float32), mask


def candidate_a(wire, motion, dist, mask):
    score = wire * motion
    high, low = np.percentile(score, [99.7, 99.0])
    moving = apply_hysteresis_threshold(score, low, high)
    # Among moving thin-line pixels within 30 px of a vessel, take the nearest to the target.
    near_vessel = ndi.distance_transform_edt(~mask) <= 30
    ys, xs = np.nonzero(moving & near_vessel)
    if len(ys) == 0:
        return None
    k = int(np.argmin(dist[ys, xs]))
    return [int(xs[k]), int(ys[k])]


def candidate_b(wire, motion):
    high, low = np.percentile(wire, [99.5, 98.5])
    mask = remove_small_objects(apply_hysteresis_threshold(wire, low, high), 40)
    skeleton = skeletonize(mask)
    neighbours = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    endpoints = skeleton & (neighbours == 1)
    h, w = skeleton.shape
    endpoints[:BORDER, :] = endpoints[-BORDER:, :] = endpoints[:, :BORDER] = endpoints[:, -BORDER:] = False
    ys, xs = np.nonzero(endpoints)
    if len(ys) == 0:
        return None
    local_motion = ndi.uniform_filter(motion, 9)
    k = int(np.argmax(local_motion[ys, xs]))
    return [int(xs[k]), int(ys[k])]


def process(case):
    dist, mask = load_geo(case["case_id"])
    out = {"case_id": case["case_id"]}
    for name, (index, reference) in FRAMES.items():
        gray = load_gray(case["clean_paths"][index])
        motion = gaussian(np.abs(gray - load_gray(case["clean_paths"][reference])), 1.5)
        wire = wire_response(gray)
        out[name] = {"A": candidate_a(wire, motion, dist, mask), "B": candidate_b(wire, motion)}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with open(DATA_DIR / "cases.jsonl") as f:
        cases = [json.loads(line) for line in f if line.strip()][: args.limit]
    with Pool(args.workers) as pool:
        tips = pool.map(process, cases, chunksize=4)
    with open(GEO_DIR / "tips_auto.json", "w") as f:
        json.dump(tips, f)
    for name in FRAMES:
        found_a = sum(t[name]["A"] is not None for t in tips)
        found_b = sum(t[name]["B"] is not None for t in tips)
        gaps = [np.hypot(*(np.subtract(t[name]["A"], t[name]["B"]))) for t in tips
                if t[name]["A"] is not None and t[name]["B"] is not None]
        print(f"{name}: A found {found_a}/{len(tips)}, B found {found_b}/{len(tips)}, "
              f"A-B distance px median {np.median(gaps):.0f}, within 25 px: {np.mean(np.array(gaps) <= 25):.2f}")
