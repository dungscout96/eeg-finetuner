from torch import nn
import torch
from .metrics import binary_classification_metric_collection, multiclass_classification_metric_collection
def generate_model_card(
    config: dict,
    finetuned_model: nn.Module,
    test_dataloader,
):
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

    model_card = {
        "model_name": config["foundation_model"].get("model_name", finetuned_model.__class__.__name__),
        "task": finetuned_model.task_info if hasattr(finetuned_model, 'task_info') else "unknown",
        "metrics": {k: v.item() for k, v in final_metrics.items()}
    }

    return model_card

def pretty_print_model_card(model_card: dict):
    """
    Pretty print the model card with == separators.

    Args:
        model_card (dict): The model card dictionary.
    """
    print("=" * 30)
    print(f"Model Name: {model_card['model_name']}")
    print(f"Task: {model_card['task']}")
    print("Performance Metrics:")
    for metric, value in model_card["metrics"].items():
        print(f"== {metric}: {value:.4f}")
    print("=" * 30)