# models/mlp.py
import torch
import torch.nn as nn
from .registry import register_model


@register_model("mlp")
class MLP(nn.Module):
    """
    Simple per-point MLP baseline (no interactions between points)
    """
    def __init__(self, input_dim=2, hidden_dim=128, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, mask=None):
        return self.net(x)
