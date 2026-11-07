"""Local (multiple-instance) model on cached patch features.

The decision seems to hinge on one small place: the wire tip relative to the branch that
leads to the target. Instead of pooling each image into a vector, score every patch
position with one tiny MLP shared across positions, then aggregate over positions
("does a bad spot exist anywhere?"). Inputs per position are the goal-map patch, the f6
patch, and the f6 - f1 change at that position, so a position can only score high if the
current frame, the path, and the recent motion look wrong *together* at that spot.

python local.py                       # all feature sets, validation only
python local.py --test vit_small_dinov2_448 max goal+f6+diff   # one pre-chosen config on test
"""
import argparse
import json

import torch
from torch import nn
import torch.nn.functional as F

from extract import DATA_DIR
from train import FEATURE_ROOT, load_data, metrics

INPUTS = {
    "f6": [("f6", 6)],
    "goal+f6": [("goal", 0), ("f6", 6)],
    "goal+f6+diff": [("goal", 0), ("f6", 6), ("diff", None)],
    "f6+diff": [("f6", 6), ("diff", None)],
}
AGGREGATES = ["max", "top8", "mean"]
EPOCHS, LR, WEIGHT_DECAY, HIDDEN = 30, 1e-3, 0.1, 64


class PatchScorer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.GELU(), nn.Dropout(0.2), nn.Linear(HIDDEN, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, aggregate):  # x: [B, P, in_dim] -> logit of retract-evidence [B]
        scores = self.mlp(self.norm(x)).squeeze(-1)  # [B, P]
        if aggregate == "max":
            pooled = scores.amax(1)
        elif aggregate == "top8":
            pooled = scores.topk(8, dim=1).values.mean(1)
        else:
            pooled = scores.mean(1)
        return pooled + self.bias


def build_inputs(patches, spec):
    """patches: [N, 7, P, D] fp16 -> [N, P, len(spec) * D] fp16, built in chunks."""
    parts = []
    for name, view in spec:
        parts.append(patches[:, 6] - patches[:, 1] if name == "diff" else patches[:, view])
    return torch.cat(parts, dim=-1)


def run_fold(x, labels, split, aggregate, seed=0):
    torch.manual_seed(seed)
    model = PatchScorer(x.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train = torch.tensor(split["train"]).cuda()
    # Positive class is RETRACT, since the scorer looks for evidence to retract.
    y = (labels == 0).float()
    for _ in range(EPOCHS):
        model.train()
        for batch in train[torch.randperm(len(train)).cuda()].split(16):
            loss = F.binary_cross_entropy_with_logits(model(x[batch].float(), aggregate), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    probs = {}
    with torch.no_grad():
        for name in ("validation", "test"):
            idx = torch.tensor(split[name]).cuda()
            p_retract = torch.cat([torch.sigmoid(model(x[b].float(), aggregate)) for b in idx.split(32)])
            probs[name] = 1 - p_retract  # back to P(advance) for metrics()
    return probs


def evaluate(x, labels, folds, aggregate, which):
    ys, ps = [], []
    for split in folds:
        probs = run_fold(x, labels, split, aggregate)
        ys.append(labels[torch.tensor(split[which]).cuda()])
        ps.append(probs[which])
    return metrics(torch.cat(ys), torch.cat(ps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features", nargs="*")
    parser.add_argument("--test", nargs=3, metavar=("FEATURES", "AGGREGATE", "INPUTS"),
                        help="Evaluate one pre-chosen config on the pooled test set")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    with open(DATA_DIR / "cases.jsonl") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    labels = torch.tensor([c["label"] for c in cases]).cuda()
    folds = [load_data(k)[1] for k in range(5)]

    if args.test:
        name, aggregate, inputs = args.test
        patches = torch.load(FEATURE_ROOT / name / "features.pt")["patches"].cuda()
        x = build_inputs(patches, INPUTS[inputs])
        del patches
        r = evaluate(x, labels, folds, aggregate, "test")
        print(f"TEST {name} {aggregate} {inputs}:", {k: round(v, 3) for k, v in r.items()})
        raise SystemExit

    names = args.features or sorted(p.name for p in FEATURE_ROOT.iterdir() if (p / "features.pt").exists())
    rows = []
    for name in names:
        patches = torch.load(FEATURE_ROOT / name / "features.pt")["patches"].cuda()
        for inputs, spec in INPUTS.items():
            x = build_inputs(patches, spec)
            for aggregate in AGGREGATES:
                r = evaluate(x, labels, folds, aggregate, "validation")
                rows.append((name, aggregate, inputs, r))
                print(f"{name:<22} {aggregate:<5} {inputs:<13} val auroc={r['auroc']:.3f} "
                      f"acc={r['overall_accuracy']:.3f} adv={r['advance_accuracy']:.3f} "
                      f"ret={r['retract_accuracy']:.3f}", flush=True)
            del x
        del patches
        torch.cuda.empty_cache()
    best = max(rows, key=lambda r: r[3]["auroc"])
    print(f"\nBEST on validation: {best[0]} {best[1]} {best[2]} auroc={best[3]['auroc']:.3f}")
