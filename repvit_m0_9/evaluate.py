"""Evaluate a saved checkpoint and optionally measure batch-one forward latency."""
import argparse
import json
import time
from pathlib import Path

import torch

from data_loader import ImagePreprocess, make_dataloader
from engine import move_batch, run_epoch
from model import WireModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--latency", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        p.error("warmup and repeats must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; run inside a GPU allocation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = WireModel(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    common = dict(root=args.data_root, fold=checkpoint["config"]["fold"], split=args.split,
                  transform=ImagePreprocess(**checkpoint["preprocessing"]),
                  num_workers=args.num_workers)
    loader = make_dataloader(batch_size=args.batch_size, **common)
    result = {"split": args.split, "fold": checkpoint["config"]["fold"],
              "checkpoint_epoch": checkpoint["epoch"], "pretrained": checkpoint["pretrained"],
              "metrics": run_epoch(model, loader, device),
              "confusion_matrix_order": "rows=true, columns=predicted; [retract, advance]"}
    if args.latency:
        single = make_dataloader(batch_size=1, **common)
        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        samples = []
        with torch.inference_mode():
            first = move_batch(next(iter(single)), device)
            for _ in range(args.warmup):
                model(first)
            synchronize()
            for batch in single:
                batch = move_batch(batch, device)
                synchronize()
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    model(batch)
                    synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
        values = torch.tensor(samples)
        result["latency"] = {
            "scope": "FP32 batch=1 full seven-image forward; excludes I/O, preprocessing and transfer",
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "mean_ms": values.mean().item(), "median_ms": values.median().item(),
            "p95_ms": values.quantile(.95).item(), "measurements": len(samples),
            "warmup": args.warmup, "repeats_per_case": args.repeats,
        }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
