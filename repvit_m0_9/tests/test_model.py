import unittest

import torch

from engine import metrics_from_confusion, run_epoch
from model import WireModel


class ModelTests(unittest.TestCase):
    def test_last_stage_updates_without_changing_earlier_layers(self):
        torch.set_num_threads(1)
        model = WireModel(pretrained=False, finetune_last_stage=True, dropout=0)
        prefix = f'encoder.stages.{len(model.encoder.stages) - 1}.'
        for name, p in model.named_parameters():
            self.assertEqual(p.requires_grad, name.startswith(prefix) or name.startswith('classifier.'))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        batch = {'goal': torch.randn(2, 3, 64, 64), 'frames': torch.randn(2, 6, 3, 64, 64),
                 'history_offsets_s': torch.zeros(2, 6), 'depth_relative_mm': torch.zeros(2, 6),
                 'label': torch.tensor([0, 1])}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        run_epoch(model, [batch], torch.device('cpu'), optimizer)
        self.assertFalse(model.encoder.stem.training)
        self.assertTrue(model.encoder.stages[-1].training)
        for key, value in model.state_dict().items():
            if key.startswith('encoder.') and not key.startswith(prefix):
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(p, before[name]) for name, p in model.named_parameters()
                            if name.startswith(prefix)))
        model.eval()
        self.assertFalse(model.encoder.stages[-1].training)

    def test_configurable_dropout_and_old_weights(self):
        model = WireModel(pretrained=False, dropout=0)
        self.assertEqual(model.classifier[2].p, 0)
        model.classifier.train()
        inputs = torch.randn(4, model.classifier[0].in_features)
        torch.testing.assert_close(model.classifier(inputs), model.classifier(inputs))
        default = WireModel(pretrained=False)
        self.assertEqual(default.classifier[2].p, .3)
        default.load_state_dict(model.state_dict())
        with self.assertRaises(ValueError):
            WireModel(pretrained=False, dropout=float('nan'))

    def test_retract_weight_scales_only_retract_gradient(self):
        class LogitsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(2, 2))

            def forward(self, batch):
                return self.logits

        gradients = []
        for weights in (None, [1.5, 1.0]):
            model = LogitsModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0)
            metrics = run_epoch(model, [{"label": torch.tensor([0, 1])}],
                                torch.device('cpu'), optimizer, class_weights=weights)
            gradients.append(model.logits.grad.clone())
            self.assertAlmostEqual(metrics['loss'], torch.log(torch.tensor(2.)).item())
            if weights is not None:
                self.assertAlmostEqual(metrics['weighted_training_loss'], 1.25 * metrics['loss'], places=6)
        torch.testing.assert_close(gradients[1][0], 1.5 * gradients[0][0])
        torch.testing.assert_close(gradients[1][1], gradients[0][1])
        with self.assertRaises(ValueError):
            run_epoch(model, [], torch.device('cpu'), class_weights=[1.5, 1.0])

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

    def test_shared_evaluation_uses_same_predictions_including_ties(self):
        from common.metrics import report
        class FixedModel(torch.nn.Module):
            def forward(self, batch):
                return batch["logits"]
        predictions = {"labels": [], "probs": []}
        batch = {"logits": torch.tensor([[0., 0.], [2., 0.], [0., 2.]]),
                 "label": torch.tensor([0, 0, 1])}
        metrics = run_epoch(FixedModel(), [batch], torch.device("cpu"),
                            prediction_sink=predictions)
        shared = report(predictions["labels"], predictions["probs"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], shared["overall_accuracy"], places=6)
        self.assertAlmostEqual(metrics["balanced_accuracy"], shared["balanced_accuracy"], places=6)

    def test_metrics(self):
        result = metrics_from_confusion(torch.tensor([[2, 1], [2, 5]]))
        self.assertAlmostEqual(result["accuracy"], .7, places=6)
        self.assertAlmostEqual(result["retract_accuracy"], 2 / 3, places=6)
        self.assertAlmostEqual(result["advance_accuracy"], 5 / 7, places=6)
        self.assertIsNone(metrics_from_confusion(torch.tensor([[0, 0], [0, 3]]))["retract_accuracy"])


if __name__ == "__main__":
    unittest.main()
