"""Synthetic checks only; no real data. Run: python -m unittest discover common/tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.data import load_all_folds, load_split  # noqa: E402
from common.metrics import auroc, report  # noqa: E402


def write_dataset(root, splits, episodes=(0, 1, 2, 3, 4, 5)):
    with open(root / "cases.jsonl", "w") as f:
        for i, episode in enumerate(episodes):
            f.write(json.dumps({"case_id": f"c{i}", "episode": episode, "label": i % 2,
                                "paths": ["x.png"] * 7, "history_offsets_s": [0] * 6,
                                "depth_relative_mm": [0] * 6}) + "\n")
    (root / "splits.json").write_text(json.dumps(splits))


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_fold_is_matched_by_id_not_position(self):
        write_dataset(self.root, [
            {"fold": 1, "train": [0, 1], "validation": [2, 3], "test": [4, 5]},
            {"fold": 0, "train": [2, 3], "validation": [4, 5], "test": [0, 1]}])
        self.assertEqual(load_split(0, self.root)[1]["test"], [0, 1])

    def test_shared_episode_is_rejected(self):
        write_dataset(self.root, [{"fold": 0, "train": [0], "validation": [1], "test": [2]}],
                      episodes=(0, 1, 0, 3, 4, 5))
        with self.assertRaises(ValueError):
            load_split(0, self.root)

    def test_pooled_folds_must_cover_every_case_once(self):
        write_dataset(self.root, [
            {"fold": 0, "train": [2, 3], "validation": [4, 5], "test": [0, 1]},
            {"fold": 1, "train": [0, 1], "validation": [4, 5], "test": [2, 3]}])
        with self.assertRaises(ValueError):
            load_all_folds(self.root)


class MetricTests(unittest.TestCase):
    def test_report(self):
        result = report([1, 1, 1, 0, 0], [0.9, 0.8, 0.4, 0.6, 0.1])
        self.assertAlmostEqual(result["overall_accuracy"], 0.6)
        self.assertAlmostEqual(result["advance_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["retract_accuracy"], 0.5)
        self.assertAlmostEqual(result["always_advance_accuracy"], 0.6)
        self.assertAlmostEqual(result["auroc"], 5 / 6)

    def test_one_class_only_gives_none_not_an_error(self):
        result = report([1, 1], [0.9, 0.2])
        self.assertAlmostEqual(result["overall_accuracy"], 0.5)
        self.assertIsNone(result["retract_accuracy"])
        self.assertIsNone(result["auroc"])

    def test_auroc_counts_ties_as_half(self):
        self.assertAlmostEqual(auroc([1, 0], [0.5, 0.5]), 0.5)


if __name__ == "__main__":
    unittest.main()
