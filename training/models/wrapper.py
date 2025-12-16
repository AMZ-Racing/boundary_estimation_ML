# models/wrapper.py
import torch.nn as nn
from .registry import get_model_class


class ModelWrapper(nn.Module):
    """
    Unified model wrapper.
    """
    def __init__(self, model_name: str, **model_kwargs):
        super().__init__()
        ModelClass = get_model_class(model_name)
        self.model = ModelClass(**model_kwargs)

    def forward(self, x, mask=None):
        return self.model(x, mask)
