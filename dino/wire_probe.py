"""Do frozen DINOv2 patch features see the guidewire at all?

python wire_probe.py --labels                         # once, CPU: per-patch label grids
python wire_probe.py --features vit_small_dinov2_448 vit_small_dinov2_448_tiles2   # GPU

Three patch-level tasks, each a logistic probe on the frozen patch token, trained on the
train group of a fold and scored on its test group, pooled over the five folds:
  wire    f6 patches ON A VESSEL: contains a thin dark line (wire/catheter) vs. does not.
          Negatives are vessel patches, so the probe cannot pass by telling vessel from
          background.
  wire-any  same positives against every other patch (the easy version, for reference).
  circle  goal.png patches: contains the drawn target circle vs. not. Control: a large
          high-contrast mark the features should certainly see.
Plus a tip test on the clicked cases: where does the patch under the clicked f6 tip rank
among all patches of that image by wire-probe score (0 = highest)?

Labels come from the Frangi thin-ridge mask used by tip.py, reduced to each patch layout as
the fraction of the patch's pixels covered. Prints aggregate numbers only.
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects
from skimage.transform import resize

from geo import DATA_DIR, GEO_DIR, load_gray
from probe import L2_GRID, fit_logreg
from tip import load_geo, wire_response
from train import FEATURE_ROOT, load_data, load_features, metrics

GRID = 32  # 448 / 14 patches per side
LAYOUTS = {1: "full", 2: "tiles2"}
LABEL_FILE = GEO_DIR / "patch_labels.npz"
WIRE_FRAC = 0.02  # a 2.5 px line crossing a patch covers ~7% at 33 px/patch, ~15% at 16 px/patch
VESSEL_FRAC = 0.10


def grid_fraction(mask, tiles):
    """Fraction of each patch covered by mask, in the patch order extract.py produces:
    tiles row-major, then GRID x GRID patches row-major inside each tile."""
    h, w = (d // tiles * tiles for d in mask.shape)
    mask = mask[:h, :w].astype(np.float32)
    out = []
    for ty in range(tiles):
        for tx in range(tiles):
            sub = mask[ty * h // tiles:(ty + 1) * h // tiles, tx * w // tiles:(tx + 1) * w // tiles]
            out.append(resize(sub, (GRID, GRID), order=1, anti_aliasing=True).ravel())
    return np.concatenate(out)


def patch_index(xy, shape, tiles):
    """Patch holding full-resolution pixel (x, y) in the same order."""
    x, y = xy
    h, w = (d // tiles for d in shape)
    ty, tx = min(int(y) // h, tiles - 1), min(int(x) // w, tiles - 1)
    py, px = min(int(y) % h * GRID // h, GRID - 1), min(int(x) % w * GRID // w, GRID - 1)
    return (ty * tiles + tx) * GRID * GRID + py * GRID + px


def thin_line_mask(gray):
    wire = wire_response(gray)
    high, low = np.percentile(wire, [99.5, 98.5])
    return remove_small_objects(apply_hysteresis_threshold(wire, low, high), 40)


def label_case(case):
    f6 = load_gray(case["clean_paths"][6])
    wire = thin_line_mask(f6)
    _, vessel = load_geo(case["case_id"])
    circle = np.abs(load_gray(case["paths"][0]) - load_gray(case["clean_paths"][0])) > 0.1
    out = {}
    for tiles, name in LAYOUTS.items():
        out[f"wire_{name}"] = grid_fraction(wire, tiles)
        out[f"vessel_{name}"] = grid_fraction(vessel, tiles)
        out[f"circle_{name}"] = grid_fraction(circle, tiles)
    return out


def make_labels(cases, workers):
    with Pool(workers) as pool:
        rows = pool.map(label_case, cases, chunksize=4)
    arrays = {key: np.stack([r[key] for r in rows]).astype(np.float16) for key in rows[0]}
    np.savez_compressed(LABEL_FILE, **arrays)
    for key, value in arrays.items():
        print(f"{key}: mean fraction {value.astype(np.float32).mean():.4f}, "
              f"patches > 0 per image {np.mean((value > 0).sum(1)):.0f}")


def probe_task(tokens, positive, candidate, folds, device):
    """tokens [416, P, D] fp16; positive/candidate [416, P] bool. Returns pooled test metrics
    at the L2 picked on pooled validation, and per-case probe scores [416, P] from the fold
    whose test group holds the case."""
    labels = torch.from_numpy(positive)
    use = torch.from_numpy(candidate)

    def gather(indices):
        idx = torch.tensor(indices)
        x = tokens[idx].float().reshape(-1, tokens.shape[-1])
        m = use[idx].reshape(-1)
        return x[m].to(device), labels[idx].reshape(-1)[m].float().to(device)

    def run(l2, group):
        all_labels, all_probs = [], []
        for split in folds:
            x_train, y_train = gather(split["train"])
            mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
            w, b = fit_logreg((x_train - mean) / std, y_train, l2)
            x_eval, y_eval = gather(split[group])
            all_labels.append(y_eval.long()); all_probs.append(torch.sigmoid(((x_eval - mean) / std) @ w + b))
        return metrics(torch.cat(all_labels), torch.cat(all_probs))

    best_l2 = max(L2_GRID, key=lambda l2: run(l2, "validation")["auroc"])
    test = run(best_l2, "test")
    # Scores for every patch, from the model of the fold that tests this case.
    scores = torch.zeros(positive.shape)
    for split in folds:
        x_train, y_train = gather(split["train"])
        mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
        w, b = fit_logreg((x_train - mean) / std, y_train, best_l2)
        for i in split["test"]:
            scores[i] = (((tokens[i].float().to(device) - mean) / std) @ w + b).cpu()
    return best_l2, test, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", action="store_true", help="build the per-patch label grids")
    parser.add_argument("--features", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with open(DATA_DIR / "cases.jsonl") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.labels:
        make_labels(cases, args.workers)
    if not args.features:
        raise SystemExit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = [load_data(k)[1] for k in range(5)]
    grids = np.load(LABEL_FILE)
    clicks = {}
    if (GEO_DIR / "tips_clicked.json").exists():
        with open(GEO_DIR / "tips_clicked.json") as f:
            clicks = {t["case_id"]: t["f6"] for t in json.load(f) if t["f6"] is not None}
    shape = tuple(np.load(GEO_DIR / f"{cases[0]['case_id']}.npz")["shape"])
    index = {case["case_id"]: i for i, case in enumerate(cases)}

    for name in args.features:
        patches, meta = load_features(name, cases)
        tiles = meta.get("tiles", 1)
        layout = LAYOUTS[tiles]
        assert patches.shape[2] == tiles * tiles * GRID * GRID, "feature layout does not match the label grid"
        wire = grids[f"wire_{layout}"].astype(np.float32)
        vessel = grids[f"vessel_{layout}"].astype(np.float32)
        circle = grids[f"circle_{layout}"].astype(np.float32)
        print(f"== {name}: {patches.shape[2]} patches per image, "
              f"{shape[1] // tiles / GRID:.0f} px per patch")
        tasks = {
            "wire (vs other vessel patches)": (patches[:, 6], wire >= WIRE_FRAC, (wire >= WIRE_FRAC) | ((wire == 0) & (vessel >= VESSEL_FRAC))),
            "wire-any (vs all patches)": (patches[:, 6], wire >= WIRE_FRAC, (wire >= WIRE_FRAC) | (wire == 0)),
            "circle (goal map control)": (patches[:, 0], circle >= WIRE_FRAC, (circle >= WIRE_FRAC) | (circle == 0)),
        }
        for task, (tokens, positive, candidate) in tasks.items():
            l2, test, scores = probe_task(tokens, positive, candidate, folds, device)
            print(f"  {task}: test AUROC {test['auroc']:.3f}  (n={test['n']}, positives "
                  f"{test['always_advance_accuracy']:.3f}, L2={l2})")
            if task.startswith("wire ("):
                ranks = []
                for case_id, xy in clicks.items():
                    i = index[case_id]
                    p = patch_index(xy, shape, tiles)
                    ranks.append((scores[i] > scores[i, p]).float().mean().item())
                if ranks:
                    r = np.array(ranks)
                    print(f"  clicked tip patch rank by wire score: median {np.median(r):.3f}, "
                          f"in top 1%: {np.mean(r <= 0.01):.2f}, top 5%: {np.mean(r <= 0.05):.2f} (n={len(r)})")
