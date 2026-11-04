import tempfile
import unittest
from pathlib import Path

import torch

from checkpoints import CheckpointWriter


class CheckpointTests(unittest.TestCase):
    def test_distinct_criteria_ties_and_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = CheckpointWriter(tmp)
            for epoch, loss, balanced in [(1, .6, .5), (2, .7, .7), (3, .65, .7), (4, .8, .6)]:
                writer.save({'epoch': epoch, 'model': {'weight': torch.tensor([epoch])},
                             'validation': {'loss': loss, 'balanced_accuracy': balanced}})
            for name, expected in [('best.pt', 1), ('best_balanced.pt', 3), ('last.pt', 4)]:
                saved = torch.load(Path(tmp) / name, weights_only=True)
                self.assertEqual(saved['epoch'], expected)
                self.assertEqual(saved['model']['weight'].item(), expected)
            self.assertFalse(list(Path(tmp).glob('*.tmp')))
