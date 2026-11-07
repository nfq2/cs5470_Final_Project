"""Compare five ResNet + RepViT fusion rules and test the validation winner."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.latency import forward_latency  # noqa: E402
from common.metrics import report  # noqa: E402
from repvit_m0_9.data_loader import ImagePreprocess, make_dataloader  # noqa: E402
from repvit_m0_9.model import WireModel as RepViTModel  # noqa: E402
from resnet.data import WireDataset, load_data  # noqa: E402
from resnet.model import WireModel as ResNetModel  # noqa: E402


def best_threshold(labels, scores):
    values = sorted(set(scores))
    candidates = [values[0] - 1e-6, values[-1] + 1e-6]
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    best = None
    for threshold in candidates:
        metrics = report(labels, scores, threshold)
        key = (metrics["balanced_accuracy"], metrics["overall_accuracy"],
               -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    return best[1], best[2]


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = (start + end) / 2 / max(1, len(values) - 1)
        for position in range(start, end + 1):
            result[order[position]] = rank
        start = end + 1
    return result


def fit_logistic(features, labels, steps=800):
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / std
    model = nn.Linear(x.shape[1], 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=0.01)
    counts = torch.bincount(y.long(), minlength=2).float()
    sample_weights = torch.tensor([len(y) / (2 * counts[int(v)]) for v in y])
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(x).squeeze(1), y, weight=sample_weights)
        loss.backward()
        optimizer.step()
    return model.eval(), mean, std


def logistic_predict(fitted, features):
    model, mean, std = fitted
    x = (torch.tensor(features, dtype=torch.float32) - mean) / std
    with torch.no_grad():
        return model(x).squeeze(1).sigmoid().tolist()


def logistic_oof(validation_features, labels, test_features):
    folds = [[] for _ in range(5)]
    for label in (0, 1):
        indices = [i for i, value in enumerate(labels) if value == label]
        for position, index in enumerate(indices):
            folds[position % len(folds)].append(index)
    oof = [0.0] * len(labels)
    all_indices = set(range(len(labels)))
    for held_out in folds:
        training = sorted(all_indices - set(held_out))
        fitted = fit_logistic([validation_features[i] for i in training],
                              [labels[i] for i in training])
        predicted = logistic_predict(fitted, [validation_features[i] for i in held_out])
        for index, probability in zip(held_out, predicted):
            oof[index] = probability
    fitted = fit_logistic(validation_features, labels)
    return oof, logistic_predict(fitted, test_features)


def collect_resnet(checkpoint, data_root, split_name, device, num_workers):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved.get("config", {})
    model = ResNetModel(dropout=config.get("dropout", 0.3),
                        finetune_layer4=config.get("finetune_layer4", False)).to(device)
    model.load_state_dict(saved["weights"])
    model.eval()
    cases, split = load_data(data_root, saved["fold"])
    dataset = WireDataset(data_root, cases, split[split_name], saved["image_size"])
    labels, probabilities = [], []
    with torch.no_grad():
        for images, history, target in DataLoader(dataset, batch_size=8,
                                                   num_workers=num_workers):
            output = model(images.to(device), history.to(device))
            labels.extend(target.tolist())
            probabilities.extend(output.softmax(1)[:, 1].cpu().tolist())
    return labels, probabilities, model, dataset


def collect_repvit(checkpoint, data_root, split_name, device, num_workers):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = RepViTModel(pretrained=False, dropout=config.get("dropout", 0.3),
                        finetune_last_stage=config.get("finetune_last_stage", False)).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    transform = ImagePreprocess(**saved["preprocessing"])
    loader = make_dataloader(data_root, split=split_name, fold=config["fold"],
                             batch_size=8, num_workers=num_workers, transform=transform)
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch)
            labels.extend(batch["label"].cpu().tolist())
            probabilities.extend(output.softmax(1)[:, 1].cpu().tolist())
    return labels, probabilities, model, loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--resnet-checkpoint", required=True)
    parser.add_argument("--repvit-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error("output-dir already exists")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; run inside a GPU allocation")
    output.mkdir(parents=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    val_y, val_r, resnet, resnet_val = collect_resnet(
        args.resnet_checkpoint, args.data_root, "validation", device, args.num_workers)
    val_v_y, val_v, repvit, repvit_val = collect_repvit(
        args.repvit_checkpoint, args.data_root, "validation", device, args.num_workers)
    test_y, test_r, _, resnet_test = collect_resnet(
        args.resnet_checkpoint, args.data_root, "test", device, args.num_workers)
    test_v_y, test_v, _, repvit_test = collect_repvit(
        args.repvit_checkpoint, args.data_root, "test", device, args.num_workers)
    if val_y != val_v_y or test_y != test_v_y:
        raise ValueError("Model loaders returned cases in different orders")

    methods = {}
    average_val = [(a + b) / 2 for a, b in zip(val_r, val_v)]
    average_test = [(a + b) / 2 for a, b in zip(test_r, test_v)]
    methods["plain_average"] = {"threshold": 0.5,
                                 "validation": report(val_y, average_val),
                                 "test_scores": average_test}

    best_weighted = None
    for step in range(21):
        alpha = step / 20
        scores = [alpha * a + (1 - alpha) * b for a, b in zip(val_r, val_v)]
        threshold, metrics = best_threshold(val_y, scores)
        key = (metrics["balanced_accuracy"], metrics["overall_accuracy"],
               -abs(alpha - 0.5))
        if best_weighted is None or key > best_weighted[0]:
            test_scores = [alpha * a + (1 - alpha) * b for a, b in zip(test_r, test_v)]
            best_weighted = (key, alpha, threshold, metrics, test_scores)
    methods["tuned_weighted_average"] = {
        "resnet_weight": best_weighted[1], "threshold": best_weighted[2],
        "validation": best_weighted[3], "test_scores": best_weighted[4]}

    minimum_val = [min(a, b) for a, b in zip(val_r, val_v)]
    minimum_test = [min(a, b) for a, b in zip(test_r, test_v)]
    threshold, metrics = best_threshold(val_y, minimum_val)
    methods["retract_veto_minimum"] = {"threshold": threshold,
                                        "validation": metrics,
                                        "test_scores": minimum_test}

    rank_val = [(a + b) / 2 for a, b in zip(ranks(val_r), ranks(val_v))]
    rank_test = [(a + b) / 2 for a, b in zip(ranks(test_r), ranks(test_v))]
    threshold, metrics = best_threshold(val_y, rank_val)
    methods["rank_average"] = {"threshold": threshold, "validation": metrics,
                                "test_scores": rank_test}

    val_features = [[torch.logit(torch.tensor(a).clamp(.001, .999)).item(),
                     torch.logit(torch.tensor(b).clamp(.001, .999)).item()]
                    for a, b in zip(val_r, val_v)]
    test_features = [[torch.logit(torch.tensor(a).clamp(.001, .999)).item(),
                      torch.logit(torch.tensor(b).clamp(.001, .999)).item()]
                     for a, b in zip(test_r, test_v)]
    logistic_val, logistic_test = logistic_oof(val_features, val_y, test_features)
    threshold, metrics = best_threshold(val_y, logistic_val)
    methods["balanced_logistic_stacker"] = {"threshold": threshold,
                                             "validation": metrics,
                                             "test_scores": logistic_test}

    selected_name = max(methods, key=lambda name: (
        methods[name]["validation"]["balanced_accuracy"],
        methods[name]["validation"]["overall_accuracy"],
        methods[name]["validation"]["auroc"]))
    selected = methods[selected_name]
    selected_test = report(test_y, selected["test_scores"], selected["threshold"])
    for method in methods.values():
        method.pop("test_scores")

    r_images, r_history, _ = resnet_val[0]
    r_images, r_history = r_images.unsqueeze(0).to(device), r_history.unsqueeze(0).to(device)
    v_batch = next(iter(make_dataloader(
        args.data_root, split="validation", fold=0, batch_size=1, num_workers=0,
        transform=repvit_val.dataset.transform)))
    v_batch = {key: value.to(device) for key, value in v_batch.items()}
    latency = forward_latency(lambda: (resnet(r_images, r_history), repvit(v_batch)))

    result = {
        "resnet_checkpoint": args.resnet_checkpoint,
        "repvit_checkpoint": args.repvit_checkpoint,
        "validation_methods": methods,
        "selected_method": selected_name,
        "selected_test": selected_test,
        "ensemble_forward_latency_mean_ms": latency["mean_ms"],
        "ensemble_forward_latency_median_ms": latency["median_ms"],
        "latency_scope": "Sequential ResNet + RepViT batch-one forwards; inputs already on GPU",
        "gpu": torch.cuda.get_device_name(),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
