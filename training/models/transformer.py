# models/transformer.py
import torch
import torch.nn as nn
from .registry import register_model


@register_model("transformer")
class VanillaTransformer(nn.Module):
    """
    Transformer with learned positional embeddings (order-sensitive)
    """
    def __init__(self,
                 input_dim=2,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 num_classes=5,
                 max_points=2048):
        super().__init__()

        self.embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_points, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        pos = torch.arange(N, device=x.device)
        pos = self.pos_embed(pos)[None, :, :]

        h = self.embed(x) + pos
        h = self.encoder(h, src_key_padding_mask=mask)
        return self.head(h)
