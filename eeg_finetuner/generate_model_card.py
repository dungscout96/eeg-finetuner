from torch import nn
import torch
import pandas as pd
from .metrics import binary_classification_metric_collection, multiclass_classification_metric_collection

"""
TODO:
- other baseline model comparisons
"""

def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flatten a nested dictionary into a single-level dictionary with dot notation keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def generate_model_card(
    config: dict,
    finetuned_model: nn.Module,
    test_dataloader,
) -> dict:
    """
    Generate a model card for the finetuned model based on its performance on the test dataset.

    Args:
        config (dict): Configuration dictionary.
        finetuned_model (nn.Module): The finetuned model.
        test_dataloader (DataLoader): DataLoader for the test dataset.
    Returns:
        dict: A model card containing model details and performance metrics.
    """
    if config["task"]["task_type"] == "classification":
        if config["task"]["num_classes"] == 2:
            metrics = binary_classification_metric_collection
        else:
            metrics = multiclass_classification_metric_collection
    else:
        metrics = None

    finetuned_model.eval()
    device = next(finetuned_model.parameters()).device
    metrics = metrics.to(device)

    for batch in test_dataloader:
        if type(batch) == dict:
            x, y = batch['x'].to(device), batch['y'].to(device)
        else:
            x, y = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            if hasattr(finetuned_model, 'backbone') and hasattr(finetuned_model, 'task_head'):
                representations = finetuned_model.backbone(x)
                representations = representations.flatten(start_dim=1)
                logits = finetuned_model.task_head(representations)
            else:
                logits = finetuned_model(x)

        y_hat = torch.argmax(logits, dim=1)
        metrics.update(y_hat, y)
    metrics = metrics.to("cpu")
    final_metrics = metrics.compute()
    test_metrics = {f"{k}": v.item() for k, v in final_metrics.items()}

    # Flatten config and add metrics
    # model_card = flatten_dict(config)
    model_card = config.copy()
    model_card["test_metrics"] = test_metrics

    return model_card
