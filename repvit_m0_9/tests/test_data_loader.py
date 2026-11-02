"""Synthetic-only tests: never require the Perlmutter dataset."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from data_loader import ImagePreprocess, ShortHorizonDataset, make_dataloader


class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        paths = []
        for i in range(7):
            name = f"image{i}.png"
            Image.new("RGB", (12, 10), (i * 30,) * 3).save(self.root / name)
            paths.append(name)
        case = dict(
            paths=paths,
            image_roles=["goal_map", "history_1", "history_2", "history_3",
                         "history_4", "history_5", "now"],
            history_offsets_s=[-5, -3, -2, -1.5, -0.5, 0],
            depth_relative_mm=[-10, -8, -6, -4, -2, 0],
        )
        self.cases = [dict(case, label=i % 2) for i in range(3)]
        (self.root / "manifest.json").write_text(json.dumps({"cases": self.cases}))
        # Nonsequential fold ID and reordered indices catch positional assumptions.
        (self.root / "splits.json").write_text(json.dumps([
            {"fold": 7, "train": [1], "validation": [2, 0], "test": [0]}
        ]))
        self.transform = ImagePreprocess((8, 8), mean=(0, 0, 0), std=(1, 1, 1))

    def test_index_mapping_and_temporal_alignment(self):
        item = ShortHorizonDataset(self.root, fold=7, transform=self.transform)[0]
        self.assertEqual(item["label"].item(), 1)
        self.assertEqual(tuple(item["frames"].shape), (6, 3, 8, 8))
        self.assertAlmostEqual(item["goal"].mean().item(), 0)
        for i, frame in enumerate(item["frames"], start=1):
            self.assertAlmostEqual(frame.mean().item(), i * 30 / 255, places=5)
        torch.testing.assert_close(item["history_offsets_s"], torch.tensor([-5., -3., -2., -1.5, -.5, 0.]))

    def test_validation_keeps_order_and_partial_batch(self):
        loader = make_dataloader(self.root, "validation", fold=7, batch_size=3,
                                 transform=self.transform)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["goal"].shape), (2, 3, 8, 8))
        self.assertEqual(batch["label"].tolist(), [0, 0])
        self.assertEqual(len(loader), 1)

    def test_invalid_fold_and_path_escape(self):
        with self.assertRaises(ValueError):
            ShortHorizonDataset(self.root, fold=0)
        dataset = ShortHorizonDataset(self.root, fold=7)
        dataset.cases[0]["paths"][0] = "../outside.png"
        with self.assertRaises(ValueError):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
