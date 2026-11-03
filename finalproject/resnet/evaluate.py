"""python evaluate.py validation   (or: python evaluate.py test)"""
import json
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

from data import WireDataset, load_data
from model import WireModel


if __name__ == "__main__":
    split_name = sys.argv[1] if len(sys.argv) > 1 else "validation"
    assert split_name in ["validation", "test"], "Choose validation or test."
    assert torch.cuda.is_available(), "Allocate a GPU before running evaluate.py."
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint = torch.load("resnet18.pt", map_location="cpu", weights_only=True)
    model = WireModel().to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    cases, split = load_data(checkpoint["fold"])
    dataset = WireDataset(cases, split[split_name], checkpoint["image_size"])
    loader = DataLoader(dataset, batch_size=8, num_workers=4)

    labels, predictions = [], []
    with torch.no_grad():
        for images, history, y in loader:
            outputs = model(images.to(device), history.to(device))
            labels.extend(y.tolist())
            predictions.extend(outputs.argmax(1).cpu().tolist())
    labels = torch.tensor(labels)
    predictions = torch.tensor(predictions)
    results = {"split": split_name, "n": len(dataset), "epoch": checkpoint["epoch"],
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
    results["gpu"] = torch.cuda.get_device_name()
    results["image_size"] = checkpoint["image_size"]
    results["precision"] = "FP32, TF32 disabled"
    print(json.dumps(results, indent=2))
    with open(split_name + "_results.json", "w") as f:
        json.dump(results, f, indent=2)
