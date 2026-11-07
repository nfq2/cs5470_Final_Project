"""Evaluate a saved checkpoint and optionally measure batch-one forward latency."""
import argparse
import json
import time
from pathlib import Path

import torch

from data_loader import ImagePreprocess, make_dataloader
from engine import move_batch, run_epoch
from model import WireModel
from history_baseline import HistoryModel, make_history_loader
from common import protocol
from common.data import load_split
from common.latency import forward_latency
from common.metrics import report  # data_loader.py puts the repository root on sys.path


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
    if Path(args.output).exists():
        p.error("output exists; choose a new filename")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; run inside a GPU allocation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    standardized = checkpoint["config"].get("protocol") == protocol.NAME
    if standardized and (device.type != "cuda" or args.batch_size != protocol.BATCH_SIZE):
        p.error("Standard baseline evaluation requires CUDA and batch size 8")
    numerical_only = checkpoint.get('model_name') == 'history_mlp'
    model = (HistoryModel(checkpoint['config']['hidden_dim'], checkpoint['config']['dropout'])
             if numerical_only else WireModel(pretrained=False,
                                history_order=checkpoint["config"].get("history_order", "time_then_depth")))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    common = dict(root=args.data_root, fold=checkpoint["config"]["fold"], split=args.split,
                  num_workers=args.num_workers)
    loader_factory = make_history_loader if numerical_only else make_dataloader
    if not numerical_only:
        common['transform'] = ImagePreprocess(**checkpoint['preprocessing'])
    loader = loader_factory(batch_size=args.batch_size, **common)
    predictions = {"labels": [], "probs": []}
    result = {"split": args.split, "fold": checkpoint["config"]["fold"],
              "model_name": checkpoint.get('model_name'),
              "checkpoint_selection": checkpoint.get("selection_criterion", "minimum unweighted validation loss (legacy)"),
              "checkpoint_epoch": checkpoint["epoch"], "pretrained": checkpoint["pretrained"],
              "metrics": run_epoch(model, loader, device, prediction_sink=predictions),
              "confusion_matrix_order": "rows=true, columns=predicted; [retract, advance]"}
    # Same numbers every model reports, including AUROC, from P(advance).
    result["shared_report"] = report(predictions["labels"], predictions["probs"])
    if standardized:
        _, supplied_split = load_split(checkpoint["config"]["fold"], args.data_root)
        first = move_batch(next(iter(loader_factory(batch_size=1, **common))), device)
        with torch.cuda.device(device):
            timing = forward_latency(lambda: model(first))
        result.update({
            "protocol": protocol.NAME, "model": "repvit_m0_9", "variant": "baseline",
            "case_indices": supplied_split[args.split], "labels": predictions["labels"],
            "advance_probabilities": predictions["probs"],
            "latency_samples_ms": timing["samples_ms"],
            "forward_latency_mean_ms": timing["mean_ms"],
            "forward_latency_median_ms": timing["median_ms"],
            "latency_warmups": timing["warmup"], "latency_repetitions": timing["repeats"],
            "latency_scope": protocol.LATENCY_SCOPE,
            "gpu": torch.cuda.get_device_name(device), "image_size": protocol.IMAGE_SIZE,
            "precision": protocol.PRECISION, **result["shared_report"],
        })
    elif args.latency:
        single = loader_factory(batch_size=1, **common)
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
            "scope": ("FP32 batch=1 numerical-history forward; excludes I/O and transfer" if numerical_only
                      else "FP32 batch=1 full seven-image forward; excludes I/O, preprocessing and transfer"),
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
