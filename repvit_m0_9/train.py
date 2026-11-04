"""Train on train only; choose checkpoint by validation loss, never test."""
import argparse
import json
import math
import random
from pathlib import Path

import timm
import torch

from data_loader import ImagePreprocess, make_dataloader
from engine import run_epoch
from model import MODEL_NAME, WireModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--retract-weight", type=float, default=1.0,
                   help="Training loss multiplier for true retract cases; advance stays 1.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--init-checkpoint", help="Optional earlier head-training checkpoint")
    p.add_argument("--random-init", action="store_true", help="Synthetic smoke tests only; disables pretrained weights")
    args = p.parse_args()
    if args.epochs < 1:
        p.error("epochs must be positive")
    if not math.isfinite(args.retract_weight) or args.retract_weight <= 0:
        p.error("retract-weight must be positive and finite")
    if args.init_checkpoint and args.random_init:
        p.error("init-checkpoint and random-init are mutually exclusive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable. Run on an allocated GPU, or use --device cpu for a smoke test.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model = WireModel(pretrained=not (args.random_init or args.init_checkpoint),
                      freeze_backbone=not args.finetune)
    cfg = model.encoder.pretrained_cfg
    preprocessing = {"size": [args.image_size, args.image_size],
                     "mean": list(cfg["mean"]), "std": list(cfg["std"])}
    pretrained = not args.random_init
    if args.init_checkpoint:
        saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if saved["config"]["fold"] != args.fold or saved["preprocessing"] != preprocessing:
            raise ValueError("Initialization checkpoint must use the same fold and preprocessing")
        model.load_state_dict(saved["model"])
        pretrained = saved["pretrained"]
    transform = ImagePreprocess(**preprocessing)
    common = dict(root=args.data_root, fold=args.fold, batch_size=args.batch_size,
                  num_workers=args.num_workers, transform=transform, seed=args.seed,
                  pin_memory=device.type == "cuda")
    train = make_dataloader(split="train", **common)
    validation = make_dataloader(split="validation", **common)
    model.fit_history_normalization(train.dataset.cases)
    model.to(device)
    groups = [{"params": model.classifier.parameters(), "lr": args.lr}]
    if args.finetune:
        groups.append({"params": model.encoder.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    config = vars(args)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    best_loss = float("inf")
    weights = [args.retract_weight, 1.0] if args.retract_weight != 1.0 else None
    print(json.dumps({"backbone_trainable": args.finetune,
                      "classifier_lr": args.lr, "backbone_lr": args.backbone_lr if args.finetune else None,
                      "class_weights_retract_advance": [args.retract_weight, 1.0],
                      "checkpoint_selection": "unweighted validation loss"}), flush=True)
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train, device, optimizer, class_weights=weights)
        valid = run_epoch(model, validation, device)
        record = {"epoch": epoch, "train": training, "validation": valid}
        print(json.dumps(record), flush=True)
        with (output / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if valid["loss"] < best_loss:
            best_loss = valid["loss"]
            torch.save({"model": model.state_dict(), "config": config,
                        "preprocessing": preprocessing, "epoch": epoch,
                        "validation": valid, "pretrained": pretrained,
                        "model_name": MODEL_NAME,
                        "versions": {"torch": str(torch.__version__), "timm": timm.__version__}},
                       output / "best.pt")
    print("Saved best validation-loss checkpoint:", output / "best.pt")


if __name__ == "__main__":
    main()
