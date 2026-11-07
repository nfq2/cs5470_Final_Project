"""Quick diagnostics on cached features: is there any signal, and where?

python probe.py                       # every feature set under features/
python probe.py vit_small_dinov2_448  # selected feature sets

Prints aggregate numbers only. Probes are L2-regularized logistic regressions, scored
by validation AUROC pooled over the 5 folds. With ~350 pooled cases the AUROC noise is
about +/-0.03, and the best of many probes is biased upward, so only treat a probe as
real signal if it clears ~0.60 and holds up on test.
"""
import json
import sys

import torch
import torch.nn.functional as F

from extract import DATA_DIR
from head import case_history
from train import FEATURE_ROOT, load_data, metrics

L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
TOP_K = 8  # patches with the largest f5 -> f6 change


def fit_logreg(x, y, l2):
    w = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    b = torch.zeros(1, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach(), b.detach()


def probe(features, labels, folds):
    """Pooled validation metrics for each L2 strength; returns the best by AUROC."""
    rows = []
    for l2 in L2_GRID:
        all_labels, all_probs = [], []
        for split in folds:
            train, val = split["train"], split["validation"]
            mean = features[train].mean(0)
            std = features[train].std(0).clamp_min(1e-6)
            x_train, x_val = (features[train] - mean) / std, (features[val] - mean) / std
            w, b = fit_logreg(x_train, labels[train].float(), l2)
            all_labels.append(labels[val])
            all_probs.append(torch.sigmoid(x_val @ w + b))
        rows.append((l2, metrics(torch.cat(all_labels), torch.cat(all_probs))))
    return max(rows, key=lambda row: row[1]["auroc"]), [round(r[1]["auroc"], 3) for r in rows]


def motion_vs_label(history, labels):
    """How often the next move continues the recent direction of motion."""
    for name, column, dead_zone in [("last interval speed (mm/s)", 4, 0.5),
                                    ("whole-window depth change (mm)", 5, 0.5)]:
        x = history[:, column]
        print(f"\n{name}: rows = recent motion, columns = next move (advance, retract)")
        for direction, mask in [("advancing", x > dead_zone), ("still", x.abs() <= dead_zone),
                                ("retracting", x < -dead_zone)]:
            adv = int((labels[mask] == 1).sum())
            ret = int((labels[mask] == 0).sum())
            share = adv / max(adv + ret, 1)
            print(f"  {direction:>10}: n={adv + ret:3d}  advance={adv:3d}  retract={ret:3d}  "
                  f"P(advance)={share:.2f}")


def feature_sets(name, history_goal):
    data = torch.load(FEATURE_ROOT / name / "features.pt")
    patches = data["patches"].cuda()  # [N, 7, P, D] FP16, order goal, f1..f6
    cls = data["cls"].cuda().float()
    mean = torch.stack([patches[i].float().mean(1) for i in range(len(patches))])
    # Patch-difference: the tokens where f6 changed most from f5, which should
    # include the wire tip (plus heartbeat and noise).
    moved = []
    for i in range(len(patches)):
        f5, f6 = patches[i, 5].float(), patches[i, 6].float()
        top = (f6 - f5).norm(dim=-1).topk(TOP_K).indices
        moved.append(torch.cat([f6[top].mean(0), (f6 - f5)[top].mean(0)]))
    moved = torch.stack(moved)
    del patches
    sets = {
        "cls goal+f6": cls[:, [0, 6]].flatten(1),
        "cls all 7": cls.flatten(1),
        "mean-patch all 7": mean.flatten(1),
        "cls diff (f6-f5, f6-f1)": torch.cat([cls[:, 6] - cls[:, 5], cls[:, 6] - cls[:, 1]], 1),
        "patch-diff top-k": moved,
    }
    sets.update({k + " +hist": torch.cat([v, history_goal], 1) for k, v in list(sets.items())})
    return sets


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    with open(DATA_DIR / "cases.jsonl") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    labels = torch.tensor([c["label"] for c in cases]).cuda()
    history = torch.stack([case_history(c) for c in cases]).cuda()
    goal = torch.tensor([[c["goal_xy"][0] / c["image_size_wh"][0],
                          c["goal_xy"][1] / c["image_size_wh"][1]] for c in cases]).cuda()
    history_goal = torch.cat([history, goal], 1)
    folds = [load_data(k)[1] for k in range(5)]
    print(f"cases={len(cases)} advance={int(labels.sum())} retract={int((labels == 0).sum())}")

    motion_vs_label(history.cpu(), labels.cpu())

    results = []
    for name, x in [("history", history), ("history + goal xy", history_goal)]:
        best, grid = probe(x, labels, folds)
        results.append(("-", name, best, grid))
    names = sys.argv[1:] or sorted(p.name for p in FEATURE_ROOT.iterdir() if (p / "features.pt").exists())
    for variant in names:
        for set_name, x in feature_sets(variant, history_goal).items():
            best, grid = probe(x, labels, folds)
            results.append((variant, set_name, best, grid))
        torch.cuda.empty_cache()

    print("\nPooled 5-fold validation (always-advance = "
          f"{results[0][2][1]['always_advance_accuracy']:.3f}); sorted by AUROC")
    print(f"{'features':<24} {'probe':<32} {'AUROC':>6} {'overall':>8} {'adv':>6} {'ret':>6}  AUROC per L2 {L2_GRID}")
    for variant, set_name, (l2, m), grid in sorted(results, key=lambda r: -r[2][1]["auroc"]):
        print(f"{variant:<24} {set_name:<32} {m['auroc']:6.3f} {m['overall_accuracy']:8.3f} "
              f"{m['advance_accuracy']:6.3f} {m['retract_accuracy']:6.3f}  {grid}")
