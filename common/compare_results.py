"""Create one comparison CSV and accuracy-versus-latency plot from pooled results."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Each architecture's pooled_test.json")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--plot", help="Optional output PNG (requires matplotlib)")
    args = parser.parse_args()
    records = [json.loads(Path(path).read_text()) for path in args.inputs]
    if any(item.get("split") != "pooled_test" for item in records):
        parser.error("every input must be a pooled_test result")
    fields = ["model", "overall_accuracy", "advance_accuracy", "retract_accuracy",
              "balanced_accuracy", "auroc", "forward_latency_median_ms"]
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: item[key] for key in fields} for item in records)
    if args.plot:
        import matplotlib.pyplot as plt

        for item in records:
            plt.scatter(item["forward_latency_median_ms"],
                        100 * item["overall_accuracy"], label=item["model"])
        plt.xlabel("Median forward latency (ms)")
        plt.ylabel("Pooled test accuracy (%)")
        plt.legend()
        plt.tight_layout()
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=200)


if __name__ == "__main__":
    main()
