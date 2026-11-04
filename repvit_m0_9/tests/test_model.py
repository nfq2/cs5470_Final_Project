import unittest

import torch

from engine import metrics_from_confusion, run_epoch
from model import WireModel


class ModelTests(unittest.TestCase):
    def test_production_epoch_changes_head_only(self):
        torch.set_num_threads(1)
        model = WireModel(pretrained=False)
        batch = {"goal": torch.randn(2, 3, 64, 64),
                 "frames": torch.randn(2, 6, 3, 64, 64),
                 "history_offsets_s": torch.zeros(2, 6),
                 "depth_relative_mm": torch.zeros(2, 6),
                 "label": torch.tensor([0, 1])}
        head_before = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        backbone_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
        run_epoch(model, [batch], torch.device('cpu'), optimizer)
        change = max((v - head_before[k]).abs().max().item()
                     for k, v in model.classifier.state_dict().items())
        self.assertGreater(change, 0)
        for k, v in model.encoder.state_dict().items():
            torch.testing.assert_close(v, backbone_before[k], rtol=0, atol=0)
        after_training = {k: v.clone() for k, v in model.state_dict().items()}
        run_epoch(model, [batch], torch.device('cpu'))
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, after_training[k], rtol=0, atol=0)

    def test_frozen_backbone_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        model = WireModel(pretrained=False)
        model.train()
        self.assertFalse(model.encoder.training)
        before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        batch = {"goal": torch.randn(2, 3, 64, 64),
                 "frames": torch.randn(2, 6, 3, 64, 64),
                 "history_offsets_s": torch.zeros(2, 6),
                 "depth_relative_mm": torch.zeros(2, 6)}
        out = model(batch)
        self.assertEqual(tuple(out.shape), (2, 2))
        out.sum().backward()
        self.assertIsNotNone(model.classifier[0].weight.grad)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        for key, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, before[key])
        model.fit_history_normalization([
            {"history_offsets_s": [0] * 6, "depth_relative_mm": [0] * 6},
            {"history_offsets_s": [0] * 6, "depth_relative_mm": [2] * 5 + [0]},
        ])
        self.assertTrue((model.history_std > 0).all())
        copy = WireModel(pretrained=False)
        copy.load_state_dict(model.state_dict())
        model.eval()
        copy.eval()
        torch.testing.assert_close(model(batch), copy(batch))

    def test_finetune_gradient(self):
        torch.set_num_threads(1)
        model = WireModel(pretrained=False, freeze_backbone=False)
        batch = {"goal": torch.randn(1, 3, 64, 64),
                 "frames": torch.randn(1, 6, 3, 64, 64),
                 "history_offsets_s": torch.zeros(1, 6),
                 "depth_relative_mm": torch.zeros(1, 6)}
        model(batch).sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))

    def test_metrics(self):
        result = metrics_from_confusion(torch.tensor([[2, 1], [2, 5]]))
        self.assertAlmostEqual(result["accuracy"], .7, places=6)
        self.assertAlmostEqual(result["retract_accuracy"], 2 / 3, places=6)
        self.assertAlmostEqual(result["advance_accuracy"], 5 / 7, places=6)
        self.assertIsNone(metrics_from_confusion(torch.tensor([[0, 0], [0, 3]]))["retract_accuracy"])


if __name__ == "__main__":
    unittest.main()
