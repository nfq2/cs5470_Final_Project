"""Accuracy from cached features (pooled over every trained fold), and live latency.

python evaluate.py --run vit_small_dinov2_448 --split validation
python evaluate.py --run vit_small_dinov2_448 --split test       # final settings only
python evaluate.py --run vit_small_dinov2_448 --latency
"""
import argparse
import json
import statistics
import time

import torch

from extract import build_backbone, encode, preprocess, read_images
from head import build_head, case_history
from train import RUN_ROOT, load_data, load_features, metrics, predict


def load_head(fold_dir, device):
    checkpoint = torch.load(fold_dir / "head.pt", map_location="cpu", weights_only=True)
    model = build_head(checkpoint["config"])
    model.load_state_dict(checkpoint["weights"])
    return model.to(device).eval(), checkpoint["config"]


def accuracy(run_dir, split_name, device):
    folds = sorted(run_dir.glob("fold*"))
    assert folds, f"No trained folds in {run_dir}"
    results, all_labels, all_probs, predictions = {}, [], [], []
    patches = None
    for fold_dir in folds:
        model, config = load_head(fold_dir, device)
        cases, split = load_data(config["fold"])
        if patches is None and not config["history_only"]:
            patches = load_features(config["features"], cases)[0].to(device)
        labels = torch.tensor([case["label"] for case in cases], device=device)
        history = torch.stack([case_history(case) for case in cases]).to(device)
        indices = torch.tensor(split[split_name], device=device)
        probs = predict(model, patches, history, indices)
        results[fold_dir.name] = metrics(labels[indices], probs)
        all_labels.append(labels[indices])
        all_probs.append(probs)
        predictions += [{"case_id": cases[i]["case_id"], "fold": config["fold"],
                         "label": cases[i]["label"], "p_advance": p}
                        for i, p in zip(split[split_name], probs.tolist())]
    results["pooled"] = metrics(torch.cat(all_labels), torch.cat(all_probs))
    results["folds"] = [d.name for d in folds]
    # Per-case predictions are for error analysis and stay on Perlmutter.
    with open(run_dir / f"{split_name}_predictions.json", "w") as f:
        json.dump(predictions, f)
    return results


def latency(run_dir, device):
    fold_dir = sorted(run_dir.glob("fold*"))[0]
    head, config = load_head(fold_dir, device)
    assert not config["history_only"], "Latency is only measured for image models."
    cases, _ = load_data(config["fold"])
    patches, meta = load_features(config["features"], cases)
    backbone = build_backbone(meta["model"], meta["size"], device)

    case = cases[0]
    frames = read_images(case).pin_memory()  # decoded frames in host memory
    history_cpu = case_history(case).unsqueeze(0)
    history = history_cpu.to(device)
    tiles = meta.get("tiles", 1)
    pixels = preprocess(frames.to(device), meta["size"], tiles)

    def tokens(x):  # [7 * tiles^2, P, D] -> [1, 7, tiles^2 * P, D], as extract.py saved them
        return encode(backbone, x)[0].reshape(1, 7, -1, backbone.embed_dim)

    def model_step():
        # Model latency: preprocessed tensors already on the GPU -> logits.
        return head(tokens(pixels), history)

    def end_to_end():
        # End-to-end latency: host frames -> GPU -> resize/normalize -> model -> decision on CPU.
        x = preprocess(frames.to(device, non_blocking=True), meta["size"], tiles)
        logits = head(tokens(x), history_cpu.to(device, non_blocking=True))
        return int(logits.argmax(1).item())

    def time_ms(step):
        times = []
        for _ in range(20):  # warm up
            step()
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return statistics.mean(times), statistics.median(times)

    with torch.no_grad():
        live = model_step()
        cached = head(patches[0:1].to(device).float(), history)
        model_mean, model_median = time_ms(model_step)
        e2e_mean, e2e_median = time_ms(end_to_end)
    return {"model_latency_mean_ms": model_mean, "model_latency_median_ms": model_median,
            "end_to_end_latency_mean_ms": e2e_mean, "end_to_end_latency_median_ms": e2e_median,
            # Live FP32 backbone vs. the FP16 cache the head was trained on.
            "live_vs_cached_max_logit_diff": (live - cached).abs().max().item(),
            "backbone": meta["model"], "image_size": meta["size"], "tiles": tiles,
            "backbone_params_m": sum(p.numel() for p in backbone.parameters()) / 1e6,
            "head_params_k": sum(p.numel() for p in head.parameters()) / 1e3,
            "gpu": torch.cuda.get_device_name(), "batch": "1 decision point (7 images)",
            "precision": "FP32, TF32 disabled", "repetitions": "20 warmup, 100 timed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run name under runs/")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--latency", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "Allocate a GPU before running evaluate.py."
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    run_dir = RUN_ROOT / args.run
    if args.latency:
        results, name = latency(run_dir, device), "latency"
    else:
        results, name = accuracy(run_dir, args.split, device), f"{args.split}_results"
    print(json.dumps(results, indent=2))
    with open(run_dir / f"{name}.json", "w") as f:
        json.dump(results, f, indent=2)
