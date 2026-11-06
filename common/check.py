"""Check the real dataset against what the shared loader assumes. Run on Perlmutter:

    python common/check.py

Prints counts and yes/no answers only: no case IDs, paths or per-case values.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.data import GROUPS, ROLES, data_root, fold_ids, load_all_folds  # noqa: E402

if __name__ == "__main__":
    root = data_root(sys.argv[1] if len(sys.argv) > 1 else None)
    cases, splits = load_all_folds(root)  # raises if a split is invalid or tests do not partition
    print("cases:", len(cases), " advance:", sum(c["label"] for c in cases))
    with open(root / "splits.json") as f:
        stored = [item["fold"] for item in json.load(f)]
    print("fold IDs in file order:", stored, " (position == ID:", stored == list(range(len(stored))), ")")
    for fold, split in zip(fold_ids(root), splits):
        print(f"fold {fold}:", {group: len(split[group]) for group in GROUPS})
    print("test groups cover every case exactly once: True")
    print("every case has image_roles in the expected order:",
          all(c.get("image_roles") == ROLES for c in cases))
    print("every case has clean_paths:", all(len(c.get("clean_paths", [])) == 7 for c in cases))
    manifest = root / "manifest.json"
    if manifest.exists():
        listed = json.loads(manifest.read_text())["cases"]
        print("manifest.json lists the same cases in the same order as cases.jsonl:",
              [c["case_id"] for c in listed] == [c["case_id"] for c in cases])
        print("manifest.json records equal cases.jsonl records:", listed == cases)
