"""Train the ResNet-18 baseline inside an allocated GPU session."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import WireDataset, get_history, load_data
from model import WireModel
from common.metrics import report

FOLD = 0
IMAGE_SIZE = 448
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.001


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--retract-weight", type=float, default=1.0)
    parser.add_argument("--finetune-layer4", action="store_true")
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--selection", choices=["val_loss", "balanced_accuracy", "auroc"],
                        default="val_loss")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.num_workers < 0:
        parser.error("epochs must be positive and num-workers nonnegative")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be between 0 (inclusive) and 1 (exclusive)")
    if args.image_size < 32 or args.retract_weight <= 0 or args.backbone_lr <= 0:
        parser.error("image-size, retract-weight and backbone-lr must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; run inside a GPU allocation")
    output = Path(args.output_dir).expanduser()
    if output.exists():
        parser.error("output-dir already exists; choose a new run directory")

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cases, split = load_data(args.data_root, args.fold)
    train_data = WireDataset(args.data_root, cases, split["train"], args.image_size)
    val_data = WireDataset(args.data_root, cases, split["validation"], args.image_size)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=args.num_workers)
    assert {cases[i]["label"] for i in split["validation"]} == {0, 1}, "Validation needs both classes."

    output.mkdir(parents=True)
    config = {**vars(args), "batch_size": BATCH_SIZE,
              "learning_rate": LEARNING_RATE, "optimizer": "Adam", "seed": 42,
              "backbone_frozen": not args.finetune_layer4, "model": "resnet18",
              "precision": "FP32; TF32 disabled",
              "torch_version": str(torch.__version__)}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    model = WireModel(pretrained=True, dropout=args.dropout,
                      finetune_layer4=args.finetune_layer4).to(device)
    history = torch.stack([get_history(cases[i]) for i in split["train"]])
    model.history_mean.copy_(history.mean(0).to(device))
    model.history_std.copy_(history.std(0, unbiased=False).clamp_min(1e-6).to(device))
    class_weights = torch.tensor([args.retract_weight, 1.0], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    parameter_groups = [{"params": model.classifier.parameters(), "lr": LEARNING_RATE}]
    if args.finetune_layer4:
        parameter_groups.append({"params": model.encoder.layer4.parameters(),
                                 "lr": args.backbone_lr})
    optimizer = torch.optim.Adam(parameter_groups)
    best_score = -float("inf")
    rows = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, history, labels in train_loader:
            images, history, labels = images.to(device), history.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, history)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        model.eval()
        val_loss, val_labels, val_probs = 0.0, [], []
        with torch.no_grad():
            for images, history, labels in val_loader:
                images, history, labels = images.to(device), history.to(device), labels.to(device)
                outputs = model(images, history)
                val_loss += loss_fn(outputs, labels).item() * len(labels)
                val_labels.extend(labels.cpu().tolist())
                val_probs.extend(outputs.softmax(1)[:, 1].cpu().tolist())

        train_loss /= len(train_data)
        val_loss /= len(val_data)
        metrics = report(val_labels, val_probs)
        val_accuracy = metrics["overall_accuracy"]
        advance_accuracy = metrics["advance_accuracy"]
        retract_accuracy = metrics["retract_accuracy"]
        balanced_accuracy = metrics["balanced_accuracy"]
        rows.append([epoch + 1, train_loss, val_loss, val_accuracy,
                     advance_accuracy, retract_accuracy, balanced_accuracy])
        print(f"Epoch {epoch + 1}: train loss={train_loss:.4f}, "
              f"val loss={val_loss:.4f}, val accuracy={val_accuracy:.3f}, "
              f"advance={advance_accuracy:.3f}, retract={retract_accuracy:.3f}, "
              f"balanced={balanced_accuracy:.3f}", flush=True)
        score = {"val_loss": -val_loss,
                 "balanced_accuracy": balanced_accuracy,
                 "auroc": metrics["auroc"]}[args.selection]
        if score > best_score:
            best_score = score
            torch.save({"weights": model.state_dict(), "fold": args.fold,
                        "image_size": args.image_size, "epoch": epoch + 1,
                        "config": config}, output / "best.pt")

    with open(output / "training.csv", "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy",
                         "val_advance_accuracy", "val_retract_accuracy", "val_balanced_accuracy"])
        writer.writerows(rows)
    print("Done. Best checkpoint by", args.selection + ":", output / "best.pt")
