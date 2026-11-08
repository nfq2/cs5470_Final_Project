"""Fit validation-only pairwise logistic stackers from saved probabilities.

The default mode never reads test JSON files.  It writes one result directory
per pair containing pooled out-of-fold validation metrics.  After all candidate
pairs have been compared, pass exactly one --selected-pair to evaluate that
preselected pair on the five held-out test groups.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import torch
from torch import nn

from common.metrics import report


FOLDS = range(5)


def parse_model(value):
    try:
        name, root = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use NAME=ARTIFACT_ROOT") from error
    if not name or not root:
        raise argparse.ArgumentTypeError("use NAME=ARTIFACT_ROOT")
    return name, Path(root)


def load_predictions(root, fold, split):
    path = root / f"fold{fold}" / f"{split}.json"
    item = json.loads(path.read_text())
    required = ("case_indices", "labels", "advance_probabilities")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    lengths = {len(item[key]) for key in required}
    if len(lengths) != 1:
        raise ValueError(f"unaligned arrays in {path}")
    if item.get("fold") != fold or item.get("split") != split:
        raise ValueError(f"fold/split metadata mismatch in {path}")
    return item, path


def aligned(first, second, first_path, second_path):
    for key in ("case_indices", "labels"):
        if first[key] != second[key]:
            raise ValueError(f"{key} mismatch: {first_path} vs {second_path}")


def log_odds(probability):
    probability = min(max(float(probability), 1e-3), 1 - 1e-3)
    return math.log(probability / (1 - probability))


def features(first, second):
    return [[log_odds(a), log_odds(b)] for a, b in zip(first, second)]


def fit_logistic(x_values, labels, steps=800):
    x = torch.tensor(x_values, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-8)
    normalized = (x - mean) / std
    model = nn.Linear(2, 1, dtype=torch.float64)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=0.01)
    counts = torch.bincount(y.long(), minlength=2).double()
    sample_weight = torch.tensor(
        [len(y) / (2 * counts[int(label)]) for label in y], dtype=torch.float64)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(normalized).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, weight=sample_weight)
        loss.backward()
        optimizer.step()
    return model.eval(), mean, std


def predict(fitted, x_values):
    model, mean, std = fitted
    x = (torch.tensor(x_values, dtype=torch.float64) - mean) / std
    with torch.no_grad():
        return model(x).squeeze(1).sigmoid().tolist()


def stratified_inner_folds(labels, parts=5):
    folds = [[] for _ in range(parts)]
    for label in (0, 1):
        indices = [index for index, value in enumerate(labels) if value == label]
        for position, index in enumerate(indices):
            folds[position % parts].append(index)
    return folds


def oof_predictions(x_values, labels):
    output = [None] * len(labels)
    all_indices = set(range(len(labels)))
    for held_out in stratified_inner_folds(labels):
        training = sorted(all_indices - set(held_out))
        fitted = fit_logistic([x_values[index] for index in training],
                              [labels[index] for index in training])
        held_scores = predict(fitted, [x_values[index] for index in held_out])
        for index, score in zip(held_out, held_scores):
            output[index] = score
    if any(score is None for score in output):
        raise RuntimeError("incomplete out-of-fold predictions")
    return output


def best_threshold(labels, scores):
    values = sorted(set(scores))
    candidates = [values[0] - 1e-12, values[-1] + 1e-12]
    candidates.extend((a + b) / 2 for a, b in zip(values, values[1:]))
    best = None
    for threshold in candidates:
        metrics = report(labels, scores, threshold)
        key = (metrics["balanced_accuracy"], metrics["overall_accuracy"],
               -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = key, threshold, metrics
    return best[1], best[2]


def fitted_parameters(fitted):
    model, mean, std = fitted
    return {
        "standardized_weights": model.weight.detach().squeeze(0).tolist(),
        "bias": float(model.bias.detach().item()),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
    }


def pair_name(first, second):
    return f"{first}__{second}"


def validation_result(first_name, first_root, second_name, second_root):
    pooled_labels = []
    pooled_scores = []
    fold_data = []
    sources = []
    for fold in FOLDS:
        first, first_path = load_predictions(first_root, fold, "validation")
        second, second_path = load_predictions(second_root, fold, "validation")
        aligned(first, second, first_path, second_path)
        labels = first["labels"]
        x_values = features(first["advance_probabilities"],
                            second["advance_probabilities"])
        scores = oof_predictions(x_values, labels)
        pooled_labels.extend(labels)
        pooled_scores.extend(scores)
        fold_data.append({"fold": fold, "n": len(labels), "labels": labels,
                          "scores": scores})
        sources.append({"fold": fold, first_name: str(first_path),
                        second_name: str(second_path)})

    threshold, pooled_metrics = best_threshold(pooled_labels, pooled_scores)
    folds = []
    for item in fold_data:
        folds.append({"fold": item["fold"],
                      **report(item["labels"], item["scores"], threshold)})
    return {
        "protocol": "midpoint_baseline_v1_pairwise_logistic_v1",
        "models": [first_name, second_name],
        "features": "clipped log-odds of P(advance) from both models",
        "training": "class-balanced logistic regression; five-way stratified OOF within each outer-fold validation set",
        "threshold_selection": "one threshold maximizing pooled OOF validation balanced accuracy; ties use overall accuracy then proximity to 0.5",
        "decision_threshold": threshold,
        "validation_oof": pooled_metrics,
        "validation_folds": folds,
        "validation_sources": sources,
        "test_status": "not read; pending comparison of all six pairs and validation-only pair selection",
        "latency_status": "pending selected-pair same-GPU forward benchmark",
    }


def evaluate_selected(first_name, first_root, second_name, second_root, threshold):
    pooled_labels = []
    pooled_scores = []
    folds = []
    parameters = []
    for fold in FOLDS:
        first_val, first_val_path = load_predictions(first_root, fold, "validation")
        second_val, second_val_path = load_predictions(second_root, fold, "validation")
        aligned(first_val, second_val, first_val_path, second_val_path)
        first_test, first_test_path = load_predictions(first_root, fold, "test")
        second_test, second_test_path = load_predictions(second_root, fold, "test")
        aligned(first_test, second_test, first_test_path, second_test_path)
        fitted = fit_logistic(
            features(first_val["advance_probabilities"],
                     second_val["advance_probabilities"]), first_val["labels"])
        test_scores = predict(
            fitted, features(first_test["advance_probabilities"],
                             second_test["advance_probabilities"]))
        labels = first_test["labels"]
        pooled_labels.extend(labels)
        pooled_scores.extend(test_scores)
        folds.append({"fold": fold, **report(labels, test_scores, threshold)})
        parameters.append({"fold": fold, **fitted_parameters(fitted)})
    return {
        "pooled_test": report(pooled_labels, pooled_scores, threshold),
        "test_folds": folds,
        "fold_stackers": parameters,
    }


def result_rows(result):
    fields = ["pair", "split", "n", "overall_accuracy", "advance_accuracy",
              "retract_accuracy", "balanced_accuracy", "auroc",
              "always_advance_accuracy", "decision_threshold",
              "ensemble_forward_latency_mean_ms",
              "ensemble_forward_latency_median_ms", "gpu"]
    latency_mean = result.get("ensemble_forward_latency_mean_ms", "")
    latency_median = result.get("ensemble_forward_latency_median_ms", "")
    gpu = result.get("gpu", "")
    rows = [{"pair": "+".join(result["models"]), "split": "validation_oof",
             **result["validation_oof"],
             "decision_threshold": result["decision_threshold"],
             "ensemble_forward_latency_mean_ms": latency_mean,
             "ensemble_forward_latency_median_ms": latency_median, "gpu": gpu}]
    if "pooled_test" in result:
        rows.append({"pair": "+".join(result["models"]), "split": "pooled_test",
                     **result["pooled_test"],
                     "decision_threshold": result["decision_threshold"],
                     "ensemble_forward_latency_mean_ms": latency_mean,
                     "ensemble_forward_latency_median_ms": latency_median,
                     "gpu": gpu})
    return fields, rows


def write_summary(path, result):
    fields, rows = result_rows(result)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(path, results):
    fields = result_rows(next(iter(results.values())))[0]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results.values(),
                             key=lambda item: item["validation_oof"]["balanced_accuracy"],
                             reverse=True):
            writer.writerows(result_rows(result)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True,
                        help="Repeat NAME=ARTIFACT_ROOT for every available model")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selected-pair",
                        help="After validation-only selection, evaluate exactly NAME1+NAME2 on test")
    parser.add_argument("--evaluate-all-tests", action="store_true",
                        help="Exploratory only: evaluate every generated pair on test")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace only this script's existing result files")
    args = parser.parse_args()
    if args.selected_pair and args.evaluate_all_tests:
        parser.error("choose selected-pair or evaluate-all-tests, not both")
    models = dict(args.model)
    if len(models) < 2:
        parser.error("provide at least two distinct models")
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for (first_name, first_root), (second_name, second_root) in itertools.combinations(
            models.items(), 2):
        name = pair_name(first_name, second_name)
        output = args.output_root / name
        if output.exists() and not args.overwrite:
            parser.error(f"refusing to overwrite {output}")
        result = validation_result(first_name, first_root, second_name, second_root)
        output.mkdir(exist_ok=True)
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        write_summary(output / "summary.csv", result)
        results[name] = result
        print(name, json.dumps(result["validation_oof"], sort_keys=True))

    if args.selected_pair:
        names = args.selected_pair.split("+")
        if len(names) != 2 or any(name not in models for name in names):
            parser.error("selected-pair must be NAME1+NAME2 using supplied model names")
        candidates = [pair_name(*names), pair_name(*reversed(names))]
        selected_name = next((name for name in candidates if name in results), None)
        if selected_name is None:
            parser.error("selected pair was not generated")
        result = results[selected_name]
        result.update(evaluate_selected(names[0], models[names[0]], names[1],
                                        models[names[1]], result["decision_threshold"]))
        result["test_status"] = "evaluated after explicit validation-only selection"
        output = args.output_root / selected_name
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        write_summary(output / "summary.csv", result)

    if args.evaluate_all_tests:
        for name, result in results.items():
            first_name, second_name = result["models"]
            result.update(evaluate_selected(
                first_name, models[first_name], second_name, models[second_name],
                result["decision_threshold"]))
            result["test_status"] = (
                "exploratory: all currently available pairs evaluated before all six candidates were available; do not use test metrics for final selection")
            output = args.output_root / name
            (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            write_summary(output / "summary.csv", result)

    write_comparison(args.output_root / "comparison.csv", results)


if __name__ == "__main__":
    main()
