# models/registry.py
MODEL_REGISTRY = {}

def register_model(name):
    """
    Decorator to register a model class.
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name):
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]
