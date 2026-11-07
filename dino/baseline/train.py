"""Train the DINOv2-S baseline under common/BASELINE_PROTOCOL.md (one fold).

python train.py --data-root $PSCRATCH/finalproject/shorthorizon2 --output-dir RUN --fold 0

The backbone is frozen and sees no augmentation, so its output for an image is the same
in every epoch. Each fold therefore runs the backbone once over its train and validation
images (FP32, TF32 off) and trains the classifier on those vectors: identical numbers to
re-encoding every epoch, at a fraction of the GPU time. evaluate.py runs the backbone live.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from data import WireDataset, get_history, load_data  # noqa: E402
from model import MODEL_NAME, WireModel  # noqa: E402
from common.metrics import report  # noqa: E402
from common.protocol import (BATCH_SIZE, CHECKPOINT_SELECTION, DROPOUT, EPOCHS,  # noqa: E402
                             FOLDS, IMAGE_SIZE, LEARNING_RATE, NAME, NUM_WORKERS,
                             OPTIMIZER, SEED, WEIGHT_DECAY, settings)


def encode_split(model, dataset, device, workers):
    """Frozen-backbone features for a whole split, in dataset order."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers)
    features, histories, labels = [], [], []
    for images, history, y in loader:
        features.append(model.encode(images.to(device)))
        histories.append(history.to(device))
        labels.append(y.to(device))
    return torch.cat(features), torch.cat(histories), torch.cat(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    if args.epochs < 1 or args.num_workers < 0 or not 0 <= args.dropout < 1:
        parser.error("epochs must be positive, num-workers nonnegative, dropout in [0, 1)")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; run inside a GPU allocation")
    output = Path(args.output_dir).expanduser()
    if output.exists():
        parser.error("output-dir already exists; choose a new run directory")

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cases, split = load_data(args.data_root, args.fold)
    train_data = WireDataset(args.data_root, cases, split["train"], args.image_size)
    val_data = WireDataset(args.data_root, cases, split["validation"], args.image_size)
    assert {cases[i]["label"] for i in split["validation"]} == {0, 1}, "Validation needs both classes."

    output.mkdir(parents=True)
    baseline = (args.fold in FOLDS and args.epochs == EPOCHS and args.image_size == IMAGE_SIZE
                and args.dropout == DROPOUT and args.num_workers == NUM_WORKERS)
    config = {**vars(args), "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
              "weight_decay": WEIGHT_DECAY, "optimizer": OPTIMIZER, "seed": SEED,
              "backbone": MODEL_NAME, "backbone_frozen": True, "model": "dinov2_small",
              "precision": "FP32; TF32 disabled", "checkpoint_selection": CHECKPOINT_SELECTION,
              "protocol": NAME if baseline else "custom_experiment",
              "protocol_settings": settings(), "torch_version": str(torch.__version__)}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    model = WireModel(args.image_size, pretrained=True, dropout=args.dropout).to(device)
    history = torch.stack([get_history(cases[i]) for i in split["train"]])
    model.history_mean.copy_(history.mean(0).to(device))
    model.history_std.copy_(history.std(0, unbiased=False).clamp_min(1e-6).to(device))
    model.eval()
    train_x, train_h, train_y = encode_split(model, train_data, device, args.num_workers)
    val_x, val_h, val_y = encode_split(model, val_data, device, args.num_workers)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    best_val_loss = float("inf")
    rows = []

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_y), generator=generator).to(device)
        train_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model.head(train_x[batch], train_h[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)

        model.eval()
        with torch.no_grad():
            outputs = model.head(val_x, val_h)
            val_loss = loss_fn(outputs, val_y).item()
            val_probs = outputs.softmax(1)[:, 1].cpu().tolist()
        train_loss /= len(train_y)
        metrics = report(val_y.cpu().tolist(), val_probs)
        rows.append([epoch + 1, train_loss, val_loss, metrics["overall_accuracy"],
                     metrics["advance_accuracy"], metrics["retract_accuracy"],
                     metrics["balanced_accuracy"]])
        print(f"Epoch {epoch + 1}: train loss={train_loss:.4f}, val loss={val_loss:.4f}, "
              f"val accuracy={metrics['overall_accuracy']:.3f}, "
              f"advance={metrics['advance_accuracy']:.3f}, retract={metrics['retract_accuracy']:.3f}, "
              f"balanced={metrics['balanced_accuracy']:.3f}", flush=True)
        if val_loss < best_val_loss:  # strictly lower: a tie keeps the earlier epoch
            best_val_loss = val_loss
            torch.save({"weights": model.state_dict(), "fold": args.fold,
                        "image_size": args.image_size, "epoch": epoch + 1,
                        "config": config}, output / "best.pt")

    with open(output / "training.csv", "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy",
                         "val_advance_accuracy", "val_retract_accuracy", "val_balanced_accuracy"])
        writer.writerows(rows)
    print("Done. Best checkpoint by validation loss:", output / "best.pt")
