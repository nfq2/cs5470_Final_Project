"""python evaluate.py validation   (or: python evaluate.py test)"""
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import WireDataset, load_data
from model import WireModel
from common.latency import forward_latency  # data.py puts the repository root on sys.path
from common.metrics import report


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

    labels, probs = [], []
    with torch.no_grad():
        for images, history, y in loader:
            outputs = model(images.to(device), history.to(device))
            labels.extend(y.tolist())
            probs.extend(outputs.softmax(1)[:, 1].cpu().tolist())  # P(advance)
    # Shared report: overall, advance, retract and balanced accuracy, AUROC, always-advance.
    results = {"model": "mobilenet_v3_small", "fold": checkpoint["fold"], "split": split_name,
               "epoch": checkpoint["epoch"], **report(labels, probs)}

    # Time one whole decision: seven images and their numeric history.
    # Inputs are already on the GPU; preprocessing and transfer are excluded.
    images, history, _ = dataset[0]
    images = images.unsqueeze(0).to(device)
    history = history.unsqueeze(0).to(device)
    latency = forward_latency(lambda: model(images, history))
    results["forward_latency_mean_ms"] = latency["mean_ms"]
    results["forward_latency_median_ms"] = latency["median_ms"]
    results["latency_scope"] = "Batch-one seven-image + history forward; excludes I/O, preprocessing, transfer and startup"
    results["latency_warmups"] = latency["warmup"]
    results["latency_repetitions"] = latency["repeats"]
    results["gpu"] = torch.cuda.get_device_name()
    results["image_size"] = checkpoint["image_size"]
    results["precision"] = "FP32, TF32 disabled"
    print(json.dumps(results, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x") as f:
        json.dump(results, f, indent=2)
