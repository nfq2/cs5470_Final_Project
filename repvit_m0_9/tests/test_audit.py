import tempfile
import unittest
from pathlib import Path

import torch

from audit import audit_inputs, batchnorm_sensitivity
from data_loader import ShortHorizonDataset
import test_data_loader


class AuditTests(unittest.TestCase):
    def setUp(self):
        test_data_loader.DataLoaderTests.setUp(self)

    def test_independent_decode_catches_repeated_frames(self):
        dataset = ShortHorizonDataset(self.root, fold=7, transform=self.transform)
        with tempfile.TemporaryDirectory() as tmp:
            report = audit_inputs(dataset, 1, Path(tmp))
            self.assertEqual(report['tensor_file_mismatches'], 0)
            self.assertEqual(report['unique_native_images'], 7)
            self.assertTrue((Path(tmp) / 'sample_0_resized_views.png').exists())

        class BrokenDataset(ShortHorizonDataset):
            def __getitem__(self, i):
                item = super().__getitem__(i)
                item['frames'] = item['frames'][0].unsqueeze(0).repeat(6, 1, 1, 1)
                return item

        broken = BrokenDataset(self.root, fold=7, transform=self.transform)
        with tempfile.TemporaryDirectory() as tmp:
            report = audit_inputs(broken, 1, Path(tmp))
            self.assertEqual(report['tensor_file_mismatches'], 5)

    def test_batchnorm_probe_preserves_state(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)
                self.dropout = torch.nn.Dropout(.8)
                self.head = torch.nn.Linear(3, 2)

            def forward(self, batch):
                return self.head(self.dropout(self.bn(batch['goal']).mean((2, 3))))

        model = Toy().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        batches = [{'goal': torch.randn(2, 3, 8, 8) + 10,
                    'label': torch.tensor([0, 1])} for _ in range(2)]
        report = batchnorm_sensitivity(model, batches, torch.device('cpu'))
        self.assertEqual(report['training_cases_checked'], 4)
        self.assertTrue(report['buffers_unchanged'])
        self.assertFalse(model.bn.training)
        self.assertTrue(model.bn.track_running_stats)
        self.assertFalse(model.dropout.training)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
