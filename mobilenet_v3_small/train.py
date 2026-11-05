"""Run this after allocating a GPU: python train.py"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import WireDataset, get_history, load_data
from model import WireModel

FOLD = 0
IMAGE_SIZE = 448
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.001


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileNet baseline matching the ResNet training setup")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.num_workers < 0:
        parser.error("epochs must be positive and num-workers nonnegative")
    FOLD, EPOCHS = args.fold, args.epochs
    output = Path(args.output_dir).expanduser()
    assert torch.cuda.is_available(), "Allocate a GPU before running train.py."
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cases, split = load_data(args.data_root, FOLD)
    train_data = WireDataset(args.data_root, cases, split["train"], IMAGE_SIZE)
    val_data = WireDataset(args.data_root, cases, split["validation"], IMAGE_SIZE)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=args.num_workers)
    assert {cases[i]["label"] for i in split["validation"]} == {0, 1}, "Validation needs both classes."

    output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
              "learning_rate": LEARNING_RATE, "optimizer": "Adam", "seed": 42,
              "dropout": 0.3, "backbone_frozen": True,
              "model": "mobilenet_v3_small", "precision": "FP32; TF32 disabled",
              "selection": "minimum validation loss",
              "torch_version": str(torch.__version__)}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    model = WireModel(pretrained=True).to(device)
    history = torch.stack([get_history(cases[i]) for i in split["train"]])
    model.history_mean.copy_(history.mean(0).to(device))
    model.history_std.copy_(history.std(0, unbiased=False).clamp_min(1e-6).to(device))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    rows = []

    for epoch in range(EPOCHS):
        model.train()
        model.encoder.eval()  # Keep the frozen encoder's BatchNorm statistics fixed.
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
        val_loss, correct = 0.0, 0
        class_correct, class_total = [0, 0], [0, 0]
        with torch.no_grad():
            for images, history, labels in val_loader:
                images, history, labels = images.to(device), history.to(device), labels.to(device)
                outputs = model(images, history)
                val_loss += loss_fn(outputs, labels).item() * len(labels)
                predictions = outputs.argmax(1)
                correct += (predictions == labels).sum().item()
                for label in [0, 1]:
                    mask = labels == label
                    class_correct[label] += (predictions[mask] == label).sum().item()
                    class_total[label] += mask.sum().item()

        train_loss /= len(train_data)
        val_loss /= len(val_data)
        val_accuracy = correct / len(val_data)
        retract_accuracy = class_correct[0] / class_total[0]
        advance_accuracy = class_correct[1] / class_total[1]
        balanced_accuracy = (retract_accuracy + advance_accuracy) / 2
        rows.append([epoch + 1, train_loss, val_loss, val_accuracy,
                     advance_accuracy, retract_accuracy, balanced_accuracy])
        print(f"Epoch {epoch + 1}: train loss={train_loss:.4f}, "
              f"val loss={val_loss:.4f}, val accuracy={val_accuracy:.3f}, "
              f"advance={advance_accuracy:.3f}, retract={retract_accuracy:.3f}, "
              f"balanced={balanced_accuracy:.3f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"weights": model.state_dict(), "fold": FOLD,
                        "image_size": IMAGE_SIZE, "epoch": epoch + 1,
                        "config": config}, output / "best.pt")

    with open(output / "training.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy",
                         "val_advance_accuracy", "val_retract_accuracy", "val_balanced_accuracy"])
        writer.writerows(rows)
    print("Done. Best validation-loss checkpoint:", output / "best.pt")
