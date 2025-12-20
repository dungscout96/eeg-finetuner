from .labram import LaBRAMBackbone

def get_foundation_model(model_name: str, **kwargs):
    if model_name == "labram":
        return LaBRAMBackbone(**kwargs)
    else:
        raise ValueError(f"Unknown foundation model: {model_name}")

__all__ = ['LaBRAMBackbone']