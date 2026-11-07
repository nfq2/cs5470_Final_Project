"""Frozen DINOv2-S with the shared seven-image feature fusion and numerical history.

The protocol's "feature vector per image" is DINOv2's CLS token (384-d for ViT-S), the
counterpart of ResNet's pooled output. The classifier is the protocol's: one 128-unit ReLU
layer with dropout over the seven concatenated vectors plus the 12 history values.
"""
import timm
import torch
from torch import nn

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"


class WireModel(nn.Module):
    def __init__(self, image_size, pretrained=False, dropout=0.3):
        super().__init__()
        if image_size % 14:
            raise ValueError("DINOv2 uses 14-pixel patches; image_size must be a multiple of 14")
        # img_size resamples DINOv2's position embeddings to the 448 patch grid.
        self.encoder = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0,
                                         img_size=image_size)
        self.feature_dim = self.encoder.embed_dim
        self.encoder.requires_grad_(False)
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

    def encode(self, images):
        """[B, 7, 3, H, W] -> [B, 7 * feature_dim] CLS tokens from the frozen backbone."""
        batch = images.shape[0]
        with torch.no_grad():
            features = self.encoder(images.flatten(0, 1))  # num_classes=0: pooled CLS token
        return features.reshape(batch, -1)

    def head(self, features, history):
        history = (history - self.history_mean) / self.history_std
        return self.classifier(torch.cat([features, history], dim=1))

    def forward(self, images, history):
        if images.ndim != 5 or tuple(images.shape[1:3]) != (7, 3):
            raise ValueError("Expected images shaped [batch, 7, 3, height, width]")
        if history.shape != (images.shape[0], 12):
            raise ValueError("Expected history shaped [batch, 12]")
        return self.head(self.encode(images), history)
