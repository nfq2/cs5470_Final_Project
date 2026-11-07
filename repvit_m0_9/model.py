"""Seven-view RepViT with a task-specific binary classifier."""
from contextlib import nullcontext

import timm
import torch
from torch import nn

MODEL_NAME = "repvit_m0_9.dist_450e_in1k"


class WireModel(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True, hidden_dim=128, dropout=0.3,
                 finetune_last_stage=False, history_order="time_then_depth"):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive)")
        self.encoder = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)
        if history_order not in ("time_then_depth", "depth_then_time"):
            raise ValueError("Invalid history order")
        self.history_order = history_order
        self.finetune_last_stage = finetune_last_stage
        self.freeze_backbone = freeze_backbone and not finetune_last_stage
        self.encoder.requires_grad_(not self.freeze_backbone and not finetune_last_stage)
        if finetune_last_stage:
            self.encoder.stages[-1].requires_grad_(True)
        self.register_buffer("history_mean", torch.zeros(12))
        self.register_buffer("history_std", torch.ones(12))
        self.classifier = nn.Sequential(
            nn.Linear(7 * self.encoder.num_features + 12, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm statistics must not change during head training.
        if self.freeze_backbone:
            self.encoder.eval()
        elif self.finetune_last_stage:
            self.encoder.eval()
            self.encoder.stages[-1].train(mode)
        return self

    def forward(self, batch):
        images = torch.cat([batch["goal"].unsqueeze(1), batch["frames"]], dim=1)
        b, views, channels, height, width = images.shape
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            features = self.encoder(images.reshape(b * views, channels, height, width))
        keys = self.history_keys()
        history = torch.cat([batch[key] for key in keys], dim=1)
        history = (history - self.history_mean) / self.history_std
        return self.classifier(torch.cat([features.reshape(b, -1), history], dim=1))

    def history_keys(self):
        return (("depth_relative_mm", "history_offsets_s") if self.history_order == "depth_then_time"
                else ("history_offsets_s", "depth_relative_mm"))

    @torch.no_grad()
    def fit_history_normalization(self, training_cases):
        values = torch.tensor([
            c[self.history_keys()[0]] + c[self.history_keys()[1]] for c in training_cases
        ], dtype=torch.float32, device=self.history_mean.device)
        if not len(values) or not torch.isfinite(values).all():
            raise ValueError("Training history must be nonempty and finite")
        self.history_mean.copy_(values.mean(0))
        std = values.std(0, unbiased=False)
        self.history_std.copy_(torch.where(std < 1e-6, torch.ones_like(std), std))
