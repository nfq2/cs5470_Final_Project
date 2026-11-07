"""Evaluate a DINOv2 baseline checkpoint with the team's shared metrics and latency protocol.

python evaluate.py --data-root ROOT --checkpoint RUN/best.pt --split test --output RUN/test.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from data import WireDataset, load_data  # noqa: E402
from model import WireModel  # noqa: E402
from common.latency import forward_latency  # noqa: E402
from common.metrics import report  # noqa: E402
from common.protocol import LATENCY_SCOPE  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if output.exists():
        parser.error("output already exists; choose a new filename")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; run inside a GPU allocation")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    model = WireModel(checkpoint["image_size"], pretrained=False,  # weights come from the checkpoint
                      dropout=config.get("dropout", 0.3)).to(device)
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
    results = {"protocol": config.get("protocol", "legacy"),
               "model": "dinov2_small", "variant": "baseline",
               "split": args.split, "fold": checkpoint["fold"], "epoch": checkpoint["epoch"],
               "checkpoint_selection": config.get("checkpoint_selection", "minimum validation loss"),
               "dropout": config.get("dropout", 0.3), "backbone": config.get("backbone"),
               "case_indices": split[args.split], "labels": labels,
               "advance_probabilities": probs,
               **report(labels, probs)}

    # Time one whole decision: seven images through the live backbone, plus the history.
    # Inputs are already on the GPU; preprocessing and transfer are excluded.
    images, history, _ = dataset[0]
    images = images.unsqueeze(0).to(device)
    history = history.unsqueeze(0).to(device)
    latency = forward_latency(lambda: model(images, history))
    results["forward_latency_mean_ms"] = latency["mean_ms"]
    results["forward_latency_median_ms"] = latency["median_ms"]
    results["latency_scope"] = LATENCY_SCOPE
    results["latency_warmups"] = latency["warmup"]
    results["latency_repetitions"] = latency["repeats"]
    results["latency_samples_ms"] = latency["samples_ms"]
    results["gpu"] = torch.cuda.get_device_name()
    results["image_size"] = checkpoint["image_size"]
    results["precision"] = "FP32, TF32 disabled"
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("case_indices", "labels", "advance_probabilities", "latency_samples_ms")},
                     indent=2))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
