"""Train the head on cached features for one fold.

python train.py --features vit_small_dinov2_448 --fold 0
python train.py --history-only --fold 0
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.data import load_split as load_data  # noqa: E402
from common.metrics import report  # noqa: E402
from head import build_head, case_history  # noqa: E402

DINO_ROOT = Path(os.environ["PSCRATCH"]) / "finalproject/dino"
FEATURE_ROOT = DINO_ROOT / "features"
RUN_ROOT = DINO_ROOT / "runs"


def metrics(labels, probs):
    """labels: 1 = advance, 0 = retract. probs: P(advance). Threshold 0.5."""
    return report(labels.tolist(), probs.tolist())


def load_features(name, cases):
    with open(FEATURE_ROOT / name / "meta.json") as f:
        meta = json.load(f)
    assert meta["case_ids"] == [case["case_id"] for case in cases], "Features are out of case order."
    return torch.load(FEATURE_ROOT / name / "features.pt")["patches"], meta


def predict(model, patches, history, indices, batch_size=32):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            x = patches[batch].float() if patches is not None else None
            probs.append(model(x, history[batch]).softmax(1)[:, 1])
    return torch.cat(probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="vit_small_dinov2_448")
    parser.add_argument("--history-only", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="Run name (default: features name, or history_only)")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "Allocate a GPU before running train.py."
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cases, split = load_data(args.fold)
    labels = torch.tensor([case["label"] for case in cases], device=device)
    history = torch.stack([case_history(case) for case in cases]).to(device)
    train_idx = torch.tensor(split["train"], device=device)
    val_idx = torch.tensor(split["validation"], device=device)

    config = {"features": None if args.history_only else args.features,
              "history_only": args.history_only, "dropout": args.dropout, "dim": None,
              "fold": args.fold, "epochs": args.epochs, "lr": args.lr,
              "weight_decay": args.weight_decay, "batch_size": args.batch_size, "seed": args.seed}
    patches = None
    if not args.history_only:
        patches, meta = load_features(args.features, cases)
        patches = patches.to(device)  # FP16 on the GPU; batches are cast to FP32.
        config["dim"] = patches.shape[-1]

    model = build_head(config).to(device)
    model.history_mean.copy_(history[train_idx].mean(0))
    model.history_std.copy_(history[train_idx].std(0, unbiased=False).clamp_min(1e-6))
    # Plain cross-entropy, no class weights or augmentation: the midpoint model is the
    # clean "before" for the retract-weighting and augmentation experiments.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    rows = []
    for epoch in range(args.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator).to(device)]
        train_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            x = patches[batch].float() if patches is not None else None
            loss = loss_fn(model(x, history[batch]), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)

        probs = predict(model, patches, history, val_idx)
        val = metrics(labels[val_idx], probs)
        val_loss = nn.functional.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6),
                                                      labels[val_idx].float()).item()
        rows.append([epoch + 1, train_loss / len(train_idx), val_loss, val["auroc"],
                     val["overall_accuracy"], val["advance_accuracy"],
                     val["retract_accuracy"], val["balanced_accuracy"]])
        print(f"Epoch {epoch + 1}: train loss={rows[-1][1]:.4f}, val loss={val_loss:.4f}, "
              f"val auroc={val['auroc']:.3f}, accuracy={val['overall_accuracy']:.3f}, "
              f"advance={val['advance_accuracy']:.3f}, retract={val['retract_accuracy']:.3f}",
              flush=True)

    # Fixed epoch count, no best-epoch picking: the count is chosen from fold-averaged
    # validation curves, so the last epoch is the checkpoint.
    out = RUN_ROOT / (args.name or ("history_only" if args.history_only else args.features)) / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": model.state_dict(), "config": config}, out / "head.pt")
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out / "training.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_auroc", "val_accuracy",
                         "val_advance_accuracy", "val_retract_accuracy", "val_balanced_accuracy"])
        writer.writerows(rows)
    print(f"Saved {out}")
