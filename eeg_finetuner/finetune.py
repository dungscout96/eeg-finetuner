import lightning as L
import torch
from torch.nn import functional as F

from .foundation_model import get_foundation_model
from .task_head import get_task_head
from .metrics import binary_classification_metric_collection, multiclass_classification_metric_collection

"""TODO: 
- add support for other eval type such as k-fold cross validation, regression metrics, etc.
- automatic hyperparameter tuning
"""

class FEMTaskAdapter(L.LightningModule):
    # https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html
    def __init__(self, 
        foundation_model: dict, 
        task: dict, 
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.backbone = get_foundation_model(**foundation_model)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.eval()

        self.task_head = get_task_head(
            task["task_type"], 
            decoder_type=task["decoder_type"], 
            num_classes=task["num_classes"],
            input_size=foundation_model["embed_dim"]
            )
        self.task_info = task
        if task["task_type"] == "classification":
            if task["num_classes"] == 2:
                self.metrics = binary_classification_metric_collection
            else:
                self.metrics = multiclass_classification_metric_collection
        else:
            self.metrics = None  # Add regression metrics if needed
        if self.metrics:
            self.train_metrics = self.metrics.clone(prefix='train_')
            self.val_metrics = self.metrics.clone(prefix='val_')

    def forward(self, x, y, *args):
        if self.freeze_backbone:
            with torch.no_grad():
                representations = self.backbone(x)
        else:
            representations = self.backbone(x)
        representations = representations.flatten(start_dim=1)

        logits = self.task_head(representations)

        if self.task_info["task_type"] == "classification":
            loss = F.cross_entropy(logits, y)
        elif self.task_info["task_type"] == "regression":
            loss = F.mse_loss(logits.squeeze(), y.float())
        
        return loss, logits

    def training_step(self, batch, batch_idx):
        if type(batch) == dict:
            loss, logits = self(**batch)
        else:
            loss, logits = self(*batch)
        self.log('train_loss', loss)
        if self.train_metrics:
            y_hat = torch.argmax(logits, dim=1)
            output = self.train_metrics(y_hat, batch[1] if type(batch) != dict else batch['y'])
            self.log_dict(output)
        return loss

    def validation_step(self, batch, batch_idx):
        if type(batch) == dict:
            loss, logits = self(**batch)
        else:
            loss, logits = self(*batch)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
