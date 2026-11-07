"""Train, evaluate and pool one architecture under the shared baseline protocol."""
import argparse
import subprocess
import sys
from pathlib import Path

from common.aggregate_folds import aggregate, write_summary
from common.protocol import FOLDS
import json


def run(command, cwd):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="Folder containing compatible train.py and evaluate.py")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not (model_dir / "train.py").is_file() or not (model_dir / "evaluate.py").is_file():
        parser.error("model-dir must contain train.py and evaluate.py")
    if output_root.exists():
        parser.error("output-root already exists; use a new directory")
    output_root.mkdir(parents=True)

    test_files = []
    for fold in FOLDS:
        fold_dir = output_root / f"fold{fold}"
        run([sys.executable, "train.py", "--data-root", data_root,
             "--output-dir", fold_dir, "--fold", fold], model_dir)
        for split in ("validation", "test"):
            destination = fold_dir / f"{split}.json"
            run([sys.executable, "evaluate.py", "--data-root", data_root,
                 "--checkpoint", fold_dir / "best.pt", "--split", split,
                 "--output", destination], model_dir)
        test_files.append(fold_dir / "test.json")

    pooled = aggregate(test_files, data_root)
    (output_root / "pooled_test.json").write_text(json.dumps(pooled, indent=2) + "\n")
    write_summary(pooled, output_root / "summary.csv")
    print("Finished all five folds. Pooled result:", output_root / "pooled_test.json")


if __name__ == "__main__":
    main()
