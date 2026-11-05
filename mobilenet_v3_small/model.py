"""Frozen MobileNetV3-Small with the ResNet baseline's fusion/classifier design."""
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class WireModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights)
        self.feature_dim = self.encoder.classifier[0].in_features
        self.encoder.classifier = nn.Identity()
        self.encoder.requires_grad_(False)
        self.register_buffer("history_mean", torch.zeros(12))
        self.register_buffer("history_std", torch.ones(12))
        self.classifier = nn.Sequential(
            nn.Linear(7 * self.feature_dim + 12, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Freeze BatchNorm statistics as well as weights.
        return self

    def forward(self, images, history):
        if images.ndim != 5 or images.shape[1:3] != (7, 3):
            raise ValueError("Expected images shaped [batch, 7, 3, height, width]")
        batch, views, channels, height, width = images.shape
        if history.shape != (batch, 12):
            raise ValueError("Expected history shaped [batch, 12]")
        with torch.no_grad():
            features = self.encoder(images.reshape(batch * views, channels, height, width))
        history = (history - self.history_mean) / self.history_std
        return self.classifier(torch.cat([features.reshape(batch, -1), history], dim=1))
