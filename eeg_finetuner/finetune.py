import lightning as L
from foundation_model import get_foundation_model
from task_head import get_task_head
import torch
from torch.nn import functional as F

class FEMTaskAdapter(L.LightningModule):
    # https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html
    def __init__(self, 
        foundation_model: dict = {
            "model_name": "labram",
            "input_size": 128,
            "num_channels": 64,
            "embedding_size": 128
        }, 
        task: dict = {
            "task_type": "classification",
            "num_classes": 2,
            "decoder_type": "linear"
        }, 
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
            input_size=foundation_model["embedding_size"]
            )
        self.task_info = task

    def forward(self, x, y, *args):
        if self.freeze_backbone:
            with torch.no_grad():
                representations = self.backbone(x)
        else:
            representations = self.backbone(x).flatten()

        y_hat = self.task_head(representations)

        if self.task_info["task_type"] == "classification":
            loss = F.cross_entropy(y_hat, y)
        elif self.task_info["task_type"] == "regression":
            loss = F.mse_loss(y_hat.squeeze(), y.float())
        
        return loss

    def training_step(self, batch, batch_idx):
        if type(batch) == dict:
            loss = self(**batch)
        else:
            loss = self(*batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if type(batch) == dict:
            loss = self(**batch)
        else:
            loss = self(*batch)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)