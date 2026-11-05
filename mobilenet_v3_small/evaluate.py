"""python evaluate.py validation   (or: python evaluate.py test)"""
import json
import statistics
import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from data import WireDataset, load_data
from model import WireModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MobileNet with ResNet-compatible timing")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Choose a new output filename to preserve prior results")
    split_name = args.split
    assert split_name in ["validation", "test"], "Choose validation or test."
    assert torch.cuda.is_available(), "Allocate a GPU before running evaluate.py."
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = WireModel().to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    cases, split = load_data(args.data_root, checkpoint["fold"])
    dataset = WireDataset(args.data_root, cases, split[split_name], checkpoint["image_size"])
    loader = DataLoader(dataset, batch_size=8, num_workers=args.num_workers)

    labels, predictions = [], []
    with torch.no_grad():
        for images, history, y in loader:
            outputs = model(images.to(device), history.to(device))
            labels.extend(y.tolist())
            predictions.extend(outputs.argmax(1).cpu().tolist())
    labels = torch.tensor(labels)
    predictions = torch.tensor(predictions)
    results = {"model": "mobilenet_v3_small", "fold": checkpoint["fold"], "split": split_name, "n": len(dataset), "epoch": checkpoint["epoch"],
               "overall_accuracy": (labels == predictions).float().mean().item(),
               "always_advance_accuracy": (labels == 1).float().mean().item()}
    for name, label in [("advance", 1), ("retract", 0)]:
        mask = labels == label
        results[name + "_accuracy"] = ((predictions[mask] == label).float().mean().item()
                                        if mask.any() else None)
    class_accuracies = [results["advance_accuracy"], results["retract_accuracy"]]
    results["balanced_accuracy"] = (sum(class_accuracies) / 2
                                    if None not in class_accuracies else None)

    # Time one whole decision: seven images and their numeric history.
    # Inputs are already on the GPU; preprocessing and transfer are excluded.
    images, history, _ = dataset[0]
    images = images.unsqueeze(0).to(device)
    history = history.unsqueeze(0).to(device)
    times = []
    with torch.no_grad():
        for _ in range(20):  # Warm up the GPU first.
            model(images, history)
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(images, history)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    results["forward_latency_mean_ms"] = statistics.mean(times)
    results["forward_latency_median_ms"] = statistics.median(times)
    results["latency_scope"] = "Batch-one seven-image + history forward; excludes I/O, preprocessing, transfer and startup"
    results["latency_warmups"] = 20
    results["latency_repetitions"] = 100
    results["gpu"] = torch.cuda.get_device_name()
    results["image_size"] = checkpoint["image_size"]
    results["precision"] = "FP32, TF32 disabled"
    print(json.dumps(results, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x") as f:
        json.dump(results, f, indent=2)
