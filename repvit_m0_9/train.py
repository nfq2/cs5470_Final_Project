"""Train on train only; choose checkpoint by validation loss, never test."""
import argparse
import json
import math
import random
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader

from data_loader import ImagePreprocess, make_dataloader
from engine import run_epoch
from model import MODEL_NAME, WireModel
from checkpoints import CheckpointWriter
from common import protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=protocol.EPOCHS)
    p.add_argument("--batch-size", type=int, default=protocol.BATCH_SIZE)
    p.add_argument("--image-size", type=int, default=protocol.IMAGE_SIZE)
    p.add_argument("--num-workers", type=int, default=protocol.NUM_WORKERS)
    p.add_argument("--lr", type=float, default=protocol.LEARNING_RATE)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=protocol.DROPOUT,
                   help="Classifier dropout probability; use 0 to disable")
    p.add_argument("--retract-weight", type=float, default=1.0,
                   help="Training loss multiplier for true retract cases; advance stays 1.0")
    p.add_argument("--seed", type=int, default=protocol.SEED)
    p.add_argument("--device", default="cuda")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--finetune", action="store_true", help="Train the entire backbone")
    scope.add_argument("--finetune-last-stage", action="store_true",
                       help="Train only the last backbone stage and classifier")
    p.add_argument("--init-checkpoint", help="Optional earlier head-training checkpoint")
    p.add_argument("--random-init", action="store_true", help="Synthetic smoke tests only; disables pretrained weights")
    p.add_argument("--experiment", action="store_true",
                   help="Opt out of the standardized baseline for custom or synthetic runs")
    args = p.parse_args()
    if not args.experiment:
        expected = dict(epochs=protocol.EPOCHS, batch_size=protocol.BATCH_SIZE,
                        image_size=protocol.IMAGE_SIZE, lr=protocol.LEARNING_RATE,
                        dropout=protocol.DROPOUT, seed=protocol.SEED, retract_weight=1.0)
        if any(getattr(args, k) != v for k, v in expected.items()) or any(
                (args.finetune, args.finetune_last_stage, args.init_checkpoint, args.random_init)):
            p.error("Baseline settings must match common.protocol; use --experiment for custom runs")
        if args.fold not in protocol.FOLDS or args.device != "cuda":
            p.error("Baseline requires a supplied fold and --device cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.epochs < 1:
        p.error("epochs must be positive")
    if not 0 <= args.dropout < 1:
        p.error("dropout must be between 0 (inclusive) and 1 (exclusive)")
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
                      freeze_backbone=not args.finetune, dropout=args.dropout,
                      finetune_last_stage=args.finetune_last_stage,
                      hidden_dim=protocol.HIDDEN_DIM,
                      history_order="time_then_depth" if args.experiment else protocol.HISTORY_ORDER)
    cfg = model.encoder.pretrained_cfg
    preprocessing = {"size": [args.image_size, args.image_size],
                     "mean": list(cfg["mean"] if args.experiment else protocol.IMAGENET_MEAN),
                     "std": list(cfg["std"] if args.experiment else protocol.IMAGENET_STD)}
    pretrained = not args.random_init
    if args.init_checkpoint:
        saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if saved["config"]["fold"] != args.fold or saved["preprocessing"] != preprocessing:
            raise ValueError("Initialization checkpoint must use the same fold and preprocessing")
        model.history_order = saved["config"].get("history_order", "time_then_depth")
        model.load_state_dict(saved["model"])
        pretrained = saved["pretrained"]
    transform = ImagePreprocess(**preprocessing)
    common = dict(root=args.data_root, fold=args.fold, batch_size=args.batch_size,
                  num_workers=args.num_workers, transform=transform, seed=args.seed,
                  pin_memory=device.type == "cuda")
    train = make_dataloader(split="train", **common)
    validation = make_dataloader(split="validation", **common)
    # Separate generator: this diagnostic pass must not change training shuffle
    # order or consume the global RNG used by dropout.
    train_eval = DataLoader(train.dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(args.seed))
    model.fit_history_normalization(train.dataset.cases)
    model.to(device)
    groups = [{"params": model.classifier.parameters(), "lr": args.lr}]
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = (torch.optim.AdamW(groups, weight_decay=1e-4) if args.experiment
                 else torch.optim.Adam(groups, lr=protocol.LEARNING_RATE, weight_decay=protocol.WEIGHT_DECAY))
    config = dict(vars(args), protocol="experiment" if args.experiment else protocol.NAME,
                  protocol_settings=protocol.settings() if not args.experiment else None,
                  history_order=model.history_order,
                  optimizer="AdamW" if args.experiment else protocol.OPTIMIZER,
                  weight_decay=1e-4 if args.experiment else protocol.WEIGHT_DECAY)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    checkpoints = CheckpointWriter(output)
    weights = [args.retract_weight, 1.0] if args.retract_weight != 1.0 else None
    print(json.dumps({"backbone_trainable": bool(backbone_params),
                      "backbone_scope": "last_stage" if args.finetune_last_stage else "all" if args.finetune else "frozen",
                      "trainable_backbone_parameters": sum(p.numel() for p in backbone_params),
                      "dropout": args.dropout,
                      "classifier_lr": args.lr, "backbone_lr": args.backbone_lr if backbone_params else None,
                      "class_weights_retract_advance": [args.retract_weight, 1.0],
                      "checkpoint_selection": "best.pt: validation loss; best_balanced.pt: validation balanced accuracy; last.pt: latest completed epoch"}), flush=True)
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train, device, optimizer, class_weights=weights)
        valid = run_epoch(model, validation, device)
        training_eval = run_epoch(model, train_eval, device)
        record = {"epoch": epoch, "train": training, "train_eval": training_eval,
                  "validation": valid}
        print(json.dumps(record), flush=True)
        with (output / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        checkpoints.save({"model": model.state_dict(), "config": config,
                        "preprocessing": preprocessing, "epoch": epoch,
                        "validation": valid, "pretrained": pretrained,
                        "model_name": MODEL_NAME,
                        "versions": {"torch": str(torch.__version__), "timm": timm.__version__}})
    print("Saved checkpoints in", output, ": best.pt, best_balanced.pt, last.pt")


if __name__ == "__main__":
    main()
