# models/set_transformer.py
import torch
import torch.nn as nn
from .registry import register_model


@register_model("set_transformer")
class SetTransformer(nn.Module):
    def __init__(self,
                 input_dim=2,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=256,
                 dropout=0.1,
                 num_classes=5):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask=None):
        h = self.embed(x)
        h = self.encoder(h, src_key_padding_mask=mask)
        return self.head(h)
