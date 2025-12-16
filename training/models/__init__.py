# models/__init__.py
from .set_transformer import SetTransformer
from .mlp import MLP
from .transformer import VanillaTransformer

from .registry import get_model_class
