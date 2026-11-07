"""The trained part: attention pooling per image, derived depth history, classifier."""
import torch
from torch import nn

NUM_VIEWS = 7  # goal map, then f1..f6
NUM_HISTORY = 7


def history_features(depths, offsets):
    """Six depths (mm) and offsets (s) -> five interval speeds, total depth change, span.

    Raw values are not used: the last depth and offset are always 0, and frame spacing
    varies between cases, so raw depth alone hides how fast the wire was moving.
    """
    dt = (offsets[..., 1:] - offsets[..., :-1]).clamp_min(1e-3)
    speeds = (depths[..., 1:] - depths[..., :-1]) / dt
    total = depths[..., -1:] - depths[..., :1]
    span = offsets[..., -1:] - offsets[..., :1]
    return torch.cat([speeds, total, span], dim=-1)


def case_history(case):
    return history_features(torch.tensor(case["depth_relative_mm"], dtype=torch.float32),
                            torch.tensor(case["history_offsets_s"], dtype=torch.float32))


class AttentionPool(nn.Module):
    """Learned queries attend over one image's patch tokens.

    The wire tip covers a handful of patches, so a plain average or the CLS token
    would wash it out. Queries start near zero, so training begins from mean pooling.
    """

    def __init__(self, dim, queries):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(queries, dim) * 0.02)

    def forward(self, tokens):  # [N, P, D] -> [N, queries * D]
        x = self.norm(tokens)
        scores = torch.einsum("qd,npd->nqp", self.queries, x) * x.shape[-1] ** -0.5
        return torch.einsum("nqp,npd->nqd", scores.softmax(-1), x).flatten(1)


class DecisionHead(nn.Module):
    def __init__(self, dim, queries=4, image_dim=64, history_dim=32, hidden=128, dropout=0.3):
        super().__init__()
        self.image_dim = image_dim
        self.pool = AttentionPool(dim, queries)
        self.project = nn.Linear(queries * dim, image_dim)
        # Training-split statistics, saved with the weights.
        self.register_buffer("history_mean", torch.zeros(NUM_HISTORY))
        self.register_buffer("history_std", torch.ones(NUM_HISTORY))
        self.history = nn.Sequential(nn.Linear(NUM_HISTORY, history_dim), nn.ReLU(),
                                     nn.Linear(history_dim, history_dim))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(NUM_VIEWS * image_dim + history_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, patches, history):
        # patches: [B, 7, P, D] in order goal, f1..f6. Concatenating keeps each image in
        # its own slot, so the goal map is never treated as an earlier time step.
        batch, views, num_patches, dim = patches.shape
        images = self.project(self.pool(patches.reshape(batch * views, num_patches, dim)))
        history = self.history((history - self.history_mean) / self.history_std)
        return self.classifier(torch.cat([images.reshape(batch, views * self.image_dim), history], 1))


class HistoryOnly(nn.Module):
    """Baseline: logistic regression on the derived depth history, no images."""

    def __init__(self):
        super().__init__()
        self.register_buffer("history_mean", torch.zeros(NUM_HISTORY))
        self.register_buffer("history_std", torch.ones(NUM_HISTORY))
        self.linear = nn.Linear(NUM_HISTORY, 2)

    def forward(self, patches, history):
        return self.linear((history - self.history_mean) / self.history_std)


def build_head(config):
    if config["history_only"]:
        return HistoryOnly()
    return DecisionHead(config["dim"], dropout=config["dropout"])
