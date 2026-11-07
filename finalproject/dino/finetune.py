"""Fine-tune DINOv2-S end to end on 2x2 tiles of the goal map, f1 and f6.

python finetune.py --fold 0                    # one fold, ~30 min on one A100
python finetune.py --summarize finetune_last4  # pooled validation/test report over folds

Why this shape: the clicked tips showed the decision depends on fine detail at the tip, and a
frozen backbone at 33 px per patch carries no usable signal. Fine-tuning tests whether the
backbone can learn to see it. Tiles keep the wire near native resolution; three images
instead of seven more than halve the cost (history-only was at chance and the next move
is independent of recent motion, so the middle frames add little).

Recipe is deliberately plain: last `--unfreeze` blocks trainable, AdamW, backbone lr 1e-5,
head lr 1e-3, fixed epochs, no augmentation, no class weights. Images are read live.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.data import data_root, load_split  # noqa: E402
from extract import MEAN, STD, build_backbone, tile  # noqa: E402
from head import NUM_HISTORY, AttentionPool, case_history  # noqa: E402
from train import RUN_ROOT, metrics  # noqa: E402

DATA_DIR = data_root()
VIEWS = [0, 1, 6]  # goal map, f1, f6 (indices into case["paths"])


class TileDataset(Dataset):
    def __init__(self, cases, indices, size, tiles):
        self.cases, self.indices, self.size, self.tiles = cases, indices, size, tiles

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        case = self.cases[self.indices[i]]
        images = []
        for view in VIEWS:
            with Image.open(DATA_DIR / case["paths"][view]) as image:
                images.append(np.asarray(image.convert("RGB")))
        x = tile(torch.from_numpy(np.stack(images)), self.tiles).permute(0, 3, 1, 2).float().div(255)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)
        x = (x - MEAN) / STD  # [views * tiles^2, 3, size, size]
        return x, case_history(case), case["label"]


class FineTuned(nn.Module):
    def __init__(self, backbone, unfreeze, views, queries=4, image_dim=64, history_dim=32, hidden=128, dropout=0.3):
        super().__init__()
        self.backbone, self.unfreeze, self.views = backbone, unfreeze, views
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        for block in backbone.blocks[len(backbone.blocks) - unfreeze:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in backbone.norm.parameters():
            parameter.requires_grad = True
        dim = backbone.embed_dim
        self.image_dim = image_dim
        self.pool = AttentionPool(dim, queries)
        self.project = nn.Linear(queries * dim, image_dim)
        self.register_buffer("history_mean", torch.zeros(NUM_HISTORY))
        self.register_buffer("history_std", torch.ones(NUM_HISTORY))
        self.history = nn.Sequential(nn.Linear(NUM_HISTORY, history_dim), nn.ReLU(), nn.Linear(history_dim, history_dim))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(views * image_dim + history_dim, hidden),
                                        nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def tokens(self, pixels):
        """Patch tokens [N, P, D]; frozen blocks run without a graph to save memory."""
        b = self.backbone
        x = b.norm_pre(b._pos_embed(b.patch_embed(pixels)))
        frozen = len(b.blocks) - self.unfreeze
        with torch.no_grad():
            for block in b.blocks[:frozen]:
                x = block(x)
        for block in b.blocks[frozen:]:
            x = block(x)
        return b.norm(x)[:, b.num_prefix_tokens:]

    def forward(self, pixels, history):
        # pixels: [B, views * tiles^2, 3, S, S]. Tiles of one view form one token sequence.
        batch, n = pixels.shape[:2]
        tokens = self.tokens(pixels.flatten(0, 1))
        tokens = tokens.reshape(batch * self.views, -1, tokens.shape[-1])
        images = self.project(self.pool(tokens)).reshape(batch, self.views * self.image_dim)
        history = self.history((history - self.history_mean) / self.history_std)
        return self.classifier(torch.cat([images, history], 1))


def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, h, y in loader:
            probs.append(model(x.to(device, non_blocking=True), h.to(device)).softmax(1)[:, 1].cpu())
            labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def summarize(name):
    out = {}
    for split in ("validation", "test"):
        probs, labels = [], []
        for fold_dir in sorted((RUN_ROOT / name).glob("fold*")):
            with open(fold_dir / f"{split}_predictions.json") as f:
                rows = json.load(f)
            probs += [r["prob"] for r in rows]; labels += [r["label"] for r in rows]
        out[split] = {"folds": len(list((RUN_ROOT / name).glob("fold*"))),
                      **metrics(torch.tensor(labels), torch.tensor(probs))}
        print(split, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out[split].items()}))
    with open(RUN_ROOT / name / "pooled_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", metavar="NAME")
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--unfreeze", type=int, default=4, help="trainable transformer blocks, from the top")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        raise SystemExit
    name = args.name or f"finetune_last{args.unfreeze}"
    assert torch.cuda.is_available(), "Allocate a GPU before running finetune.py."
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cases, split = load_split(args.fold)
    loaders = {group: DataLoader(TileDataset(cases, split[group], args.size, args.tiles),
                                 batch_size=args.batch_size, shuffle=(group == "train"),
                                 num_workers=args.workers, pin_memory=True, persistent_workers=True,
                                 generator=torch.Generator().manual_seed(args.seed))
               for group in ("train", "validation", "test")}

    backbone = build_backbone(args.model, args.size, device)
    model = FineTuned(backbone, args.unfreeze, len(VIEWS), dropout=args.dropout).to(device)
    history = torch.stack([case_history(cases[i]) for i in split["train"]])
    model.history_mean.copy_(history.mean(0).to(device))
    model.history_std.copy_(history.std(0, unbiased=False).clamp_min(1e-6).to(device))
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW([{"params": backbone_params, "lr": args.backbone_lr},
                                   {"params": head_params, "lr": args.lr}], weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    config = {**vars(args), "name": name, "views": VIEWS, "trainable_params_m": sum(p.numel() for p in backbone_params + head_params) / 1e6}
    print(json.dumps(config), flush=True)

    rows = []
    for epoch in range(args.epochs):
        model.train()
        model.backbone.eval()  # no dropout/drop-path in the backbone; DINOv2 has no BatchNorm
        train_loss, seen = 0.0, 0
        for x, h, y in loaders["train"]:
            x, h, y = x.to(device, non_blocking=True), h.to(device), y.to(device)
            loss = loss_fn(model(x, h), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y); seen += len(y)
        probs, labels = predict(model, loaders["validation"], device)
        val = metrics(labels, probs)
        rows.append([epoch + 1, train_loss / seen, val["auroc"], val["overall_accuracy"],
                     val["advance_accuracy"], val["retract_accuracy"], val["balanced_accuracy"]])
        print(f"Epoch {epoch + 1}: train loss={rows[-1][1]:.4f}, val auroc={val['auroc']:.3f}, "
              f"accuracy={val['overall_accuracy']:.3f}, advance={val['advance_accuracy']:.3f}, "
              f"retract={val['retract_accuracy']:.3f}", flush=True)

    out = RUN_ROOT / name / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    # Fixed epoch count: the last epoch is the model. Test predictions are written now so
    # pooling needs no GPU, but they are only looked at once, pooled, after validation.
    for group in ("validation", "test"):
        probs, labels = predict(model, loaders[group], device)
        with open(out / f"{group}_predictions.json", "w") as f:
            json.dump([{"index": i, "label": int(l), "prob": float(p)}
                       for i, l, p in zip(split[group], labels.tolist(), probs.tolist())], f)
    torch.save({"weights": {k: v for k, v in model.state_dict().items()
                            if not k.startswith("backbone.") or any(k.startswith(f"backbone.blocks.{j}.") for j in range(len(backbone.blocks) - args.unfreeze, len(backbone.blocks))) or k.startswith("backbone.norm.")},
                "config": config}, out / "model.pt")
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out / "training.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_auroc", "val_accuracy", "val_advance_accuracy",
                         "val_retract_accuracy", "val_balanced_accuracy"])
        writer.writerows(rows)
    print(f"Saved {out}")
