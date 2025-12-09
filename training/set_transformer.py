import torch
import torch.nn as nn

class SetTransformerClassifier(nn.Module):
    def __init__(self,
                 dim_input: int = 2,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 num_classes: int = 5):
        super().__init__()
        self.input_embed = nn.Sequential(
            nn.Linear(dim_input, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # batch_first=True for (B, N, D)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: (B, N, 2)
        mask: (B, N) boolean where True indicates PAD pod
        returns logits: (B, N, C)
        """
        # embed
        h = self.input_embed(x)  # (B, N, D)

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = mask  #(B, N) bool

        h = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B, N, D)

        logits = self.classifier(h)  # (B, N, C)
        return logits