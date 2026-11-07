"""Pool five held-out test prediction files into one 416-case report."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from common.data import load_all_folds
from common.metrics import report
from common.protocol import FOLDS, NAME


def aggregate(paths, data_root):
    cases, splits = load_all_folds(data_root)
    expected = {split["fold"]: split["test"] for split in splits}
    records = [json.loads(Path(path).read_text()) for path in paths]
    by_fold = {record.get("fold"): record for record in records}
    if len(by_fold) != len(records) or set(by_fold) != set(FOLDS):
        raise ValueError(f"Need exactly one result for each fold {list(FOLDS)}")

    model = records[0].get("model")
    labels, probabilities, indices, latency_samples = [], [], [], []
    gpus, image_sizes, precisions = set(), set(), set()
    for fold in FOLDS:
        item = by_fold[fold]
        if item.get("model") != model or item.get("split") != "test":
            raise ValueError("All inputs must be test results from the same model")
        if item.get("protocol") != NAME:
            raise ValueError(f"Fold {fold} was not run with protocol {NAME}")
        if item.get("case_indices") != expected[fold]:
            raise ValueError(f"Fold {fold} case indices do not match the supplied test split")
        fold_labels = item.get("labels", [])
        fold_probs = item.get("advance_probabilities", [])
        if not (len(fold_labels) == len(fold_probs) == len(expected[fold])):
            raise ValueError(f"Fold {fold} prediction arrays have the wrong length")
        indices.extend(expected[fold])
        labels.extend(fold_labels)
        probabilities.extend(fold_probs)
        latency_samples.extend(item.get("latency_samples_ms", []))
        gpus.add(item.get("gpu"))
        image_sizes.add(item.get("image_size"))
        precisions.add(item.get("precision"))

    if sorted(indices) != list(range(len(cases))):
        raise ValueError("The pooled test results do not cover every case exactly once")
    if len(gpus) != 1 or len(image_sizes) != 1 or len(precisions) != 1:
        raise ValueError("GPU, image size and precision must match across all folds")
    if not latency_samples:
        raise ValueError("No standardized latency samples were saved")

    return {
        "protocol": NAME,
        "model": model,
        "variant": "baseline",
        "folds": list(FOLDS),
        "split": "pooled_test",
        **report(labels, probabilities),
        "forward_latency_mean_ms": statistics.mean(latency_samples),
        "forward_latency_median_ms": statistics.median(latency_samples),
        "latency_measurements": len(latency_samples),
        "gpu": gpus.pop(),
        "image_size": image_sizes.pop(),
        "precision": precisions.pop(),
    }


def write_summary(result, path):
    fields = [
        "protocol", "model", "variant", "split", "n", "overall_accuracy",
        "advance_accuracy", "retract_accuracy", "balanced_accuracy", "auroc",
        "always_advance_accuracy", "forward_latency_mean_ms",
        "forward_latency_median_ms", "latency_measurements", "image_size",
        "precision", "gpu",
    ]
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: result[field] for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="The five fold test JSON files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    output, summary = Path(args.output), Path(args.summary)
    if output.exists() or summary.exists():
        parser.error("output and summary must be new files")
    result = aggregate(args.inputs, args.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    write_summary(result, summary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
