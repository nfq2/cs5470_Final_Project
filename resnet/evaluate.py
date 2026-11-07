"""Evaluate a checkpoint with the team's shared metrics and latency protocol."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import WireDataset, load_data
from model import WireModel
from common.latency import forward_latency  # data.py puts the repository root on sys.path
from common.metrics import report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    output = Path(args.output).expanduser()
    if output.exists():
        parser.error("output already exists; choose a new filename")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; run inside a GPU allocation")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dropout = checkpoint.get("config", {}).get("dropout", 0.3)
    finetune_layer4 = checkpoint.get("config", {}).get("finetune_layer4", False)
    model = WireModel(dropout=dropout, finetune_layer4=finetune_layer4).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    cases, split = load_data(args.data_root, checkpoint["fold"])
    dataset = WireDataset(args.data_root, cases, split[args.split], checkpoint["image_size"])
    loader = DataLoader(dataset, batch_size=8, num_workers=args.num_workers)

    labels, probs = [], []
    with torch.no_grad():
        for images, history, y in loader:
            outputs = model(images.to(device), history.to(device))
            labels.extend(y.tolist())
            probs.extend(outputs.softmax(1)[:, 1].cpu().tolist())  # P(advance)
    # Shared report: overall, advance, retract and balanced accuracy, AUROC, always-advance.
    results = {"model": "resnet18", "split": args.split, "fold": checkpoint["fold"],
               "epoch": checkpoint["epoch"],
               "checkpoint_selection": checkpoint.get("config", {}).get(
                   "selection", "val_loss"),
               "dropout": dropout,
               "retract_weight": checkpoint.get("config", {}).get("retract_weight", 1.0),
               "finetune_layer4": finetune_layer4,
               **report(labels, probs)}

    # Time one whole decision: seven images and their numeric history.
    # Inputs are already on the GPU; preprocessing and transfer are excluded.
    images, history, _ = dataset[0]
    images = images.unsqueeze(0).to(device)
    history = history.unsqueeze(0).to(device)
    latency = forward_latency(lambda: model(images, history))
    results["forward_latency_mean_ms"] = latency["mean_ms"]
    results["forward_latency_median_ms"] = latency["median_ms"]
    results["latency_scope"] = ("Batch-one seven-image + history forward; excludes I/O, "
                                "preprocessing, transfer and startup")
    results["latency_warmups"] = latency["warmup"]
    results["latency_repetitions"] = latency["repeats"]
    results["gpu"] = torch.cuda.get_device_name()
    results["image_size"] = checkpoint["image_size"]
    results["precision"] = "FP32, TF32 disabled"
    print(json.dumps(results, indent=2))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
