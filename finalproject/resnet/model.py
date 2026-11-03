"""ResNet-18 extracts image features; a small classifier predicts the action."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class WireModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()  # Get 512 features instead of ImageNet classes.
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        # Save training-data normalization with the model.
        self.register_buffer("history_mean", torch.zeros(12))
        self.register_buffer("history_std", torch.ones(12))
        self.classifier = nn.Sequential(
            nn.Linear(7 * 512 + 12, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, images, history):
        batch, views, channels, height, width = images.shape
        images = images.reshape(batch * views, channels, height, width)
        features = self.encoder(images).reshape(batch, 7 * 512)
        history = (history - self.history_mean) / self.history_std
        return self.classifier(torch.cat([features, history], dim=1))
