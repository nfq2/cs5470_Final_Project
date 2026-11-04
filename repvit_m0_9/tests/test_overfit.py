import unittest

import torch

from model import WireModel
from overfit import fit_tiny_batch, score


class OverfitTests(unittest.TestCase):
    def test_updates_both_parts_and_reports_final_step(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        model = WireModel(pretrained=False, freeze_backbone=False)
        batch = {'goal': torch.randn(2, 3, 64, 64),
                 'frames': torch.randn(2, 6, 3, 64, 64),
                 'history_offsets_s': torch.zeros(2, 6),
                 'depth_relative_mm': torch.zeros(2, 6),
                 'label': torch.tensor([0, 1])}
        report = fit_tiny_batch(model, batch, steps=2, emit=lambda line: None)
        for group in ('encoder', 'classifier'):
            self.assertGreater(report['first_update'][group + '_max_weight_change'], 0)
            self.assertGreater(report['first_update'][group + '_gradient_l2'], 0)
        self.assertEqual([r['step'] for r in report['history']], [0, 2])
        before = {k: v.clone() for k, v in model.state_dict().items()}
        score(model, batch, batch_statistics=True)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
