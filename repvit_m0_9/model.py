"""Seven-view RepViT with a task-specific binary classifier."""
from contextlib import nullcontext

import timm
import torch
from torch import nn

MODEL_NAME = "repvit_m0_9.dist_450e_in1k"


class WireModel(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True, hidden_dim=128):
        super().__init__()
        self.encoder = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)
        self.freeze_backbone = freeze_backbone
        self.encoder.requires_grad_(not freeze_backbone)
        self.register_buffer("history_mean", torch.zeros(12))
        self.register_buffer("history_std", torch.ones(12))
        self.classifier = nn.Sequential(
            nn.Linear(7 * self.encoder.num_features + 12, hidden_dim),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, 2),
        )
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm statistics must not change during head training.
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, batch):
        images = torch.cat([batch["goal"].unsqueeze(1), batch["frames"]], dim=1)
        b, views, channels, height, width = images.shape
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            features = self.encoder(images.reshape(b * views, channels, height, width))
        history = torch.cat([batch["history_offsets_s"], batch["depth_relative_mm"]], dim=1)
        history = (history - self.history_mean) / self.history_std
        return self.classifier(torch.cat([features.reshape(b, -1), history], dim=1))

    @torch.no_grad()
    def fit_history_normalization(self, training_cases):
        values = torch.tensor([
            c["history_offsets_s"] + c["depth_relative_mm"] for c in training_cases
        ], dtype=torch.float32, device=self.history_mean.device)
        if not len(values) or not torch.isfinite(values).all():
            raise ValueError("Training history must be nonempty and finite")
        self.history_mean.copy_(values.mean(0))
        std = values.std(0, unbiased=False)
        self.history_std.copy_(torch.where(std < 1e-6, torch.ones_like(std), std))
