"""Frozen ResNet-18 with seven-image feature fusion and numerical history."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class WireModel(nn.Module):
    def __init__(self, pretrained=False, dropout=0.3, finetune_layer4=False):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive)")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = resnet18(weights=weights)
        self.feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.encoder.requires_grad_(False)
        self.finetune_layer4 = finetune_layer4
        if finetune_layer4:
            self.encoder.layer4.requires_grad_(True)

        # Save training-data normalization with the model.
        self.register_buffer("history_mean", torch.zeros(12))
        self.register_buffer("history_std", torch.ones(12))
        self.classifier = nn.Sequential(
            nn.Linear(7 * self.feature_dim + 12, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images, history):
        if images.ndim != 5 or tuple(images.shape[1:3]) != (7, 3):
            raise ValueError("Expected images shaped [batch, 7, 3, height, width]")
        batch, views, channels, height, width = images.shape
        if history.shape != (batch, 12):
            raise ValueError("Expected history shaped [batch, 12]")
        with torch.set_grad_enabled(self.finetune_layer4 and self.training):
            features = self.encoder(images.reshape(batch * views, channels, height, width))
        history = (history - self.history_mean) / self.history_std
        return self.classifier(torch.cat([features.reshape(batch, -1), history], dim=1))
