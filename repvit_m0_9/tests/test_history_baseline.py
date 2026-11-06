import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from history_baseline import HistoryDataset, HistoryModel
from engine import run_epoch


class HistoryBaselineTests(unittest.TestCase):
    def test_no_images_and_split_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [dict(label=i % 2, episode=i, paths=['unused.png'] * 7,
                          history_offsets_s=[-5,-4,-3,-2,-1,0],
                          depth_relative_mm=[-i,-i,-i,-i,-i,0]) for i in range(3)]
            (root/'cases.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cases))
            (root/'splits.json').write_text(json.dumps([{'fold': 2, 'train': [1], 'validation': [2], 'test': [0]}]))
            with patch('PIL.Image.open', side_effect=AssertionError('Image opened')):
                dataset = HistoryDataset(root, fold=2)
                item = dataset[0]
            self.assertEqual(item['label'].item(), 1)
            self.assertEqual(set(item), {'history_offsets_s', 'depth_relative_mm', 'label'})
            torch.testing.assert_close(item['depth_relative_mm'], torch.tensor([-1.,-1.,-1.,-1.,-1.,0.]))

    def test_normalization_checkpoint_and_updates(self):
        model = HistoryModel(dropout=0)
        training = [dict(history_offsets_s=[-5,-4,-3,-2,-1,0], depth_relative_mm=[v]*5+[0]) for v in (-2, 2)]
        model.fit_history_normalization(training)
        self.assertEqual(model.history_mean[-2].item(), 0)
        self.assertEqual(model.history_std[-2].item(), 2)
        self.assertEqual(model.history_std[-1].item(), 1)
        batch = {'history_offsets_s': torch.tensor([c['history_offsets_s'] for c in training], dtype=torch.float32),
                 'depth_relative_mm': torch.tensor([c['depth_relative_mm'] for c in training], dtype=torch.float32),
                 'label': torch.tensor([0,1])}
        before = model.classifier[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        run_epoch(model, [batch], torch.device('cpu'), optimizer)
        self.assertFalse(torch.equal(before, model.classifier[0].weight))
        mean = model.history_mean.clone()
        batch['depth_relative_mm'] += 1000
        run_epoch(model, [batch], torch.device('cpu'))
        torch.testing.assert_close(model.history_mean, mean)
        restored = HistoryModel(dropout=0)
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        torch.testing.assert_close(restored(batch), model(batch))
