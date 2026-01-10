from copy import deepcopy

import lightning as L
import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler as torch_lr_scheduler

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
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
        label_smoothing: float = 0.0,
        task_head_overrides: dict | None = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.optimizer_config = optimizer_config or {}
        self.scheduler_config = scheduler_config or {}
        self.label_smoothing = label_smoothing if task.get("task_type") == "classification" else 0.0
        self.backbone = get_foundation_model(**foundation_model)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.eval()

        head_kwargs = dict(task.get("head") or {})
        if task_head_overrides:
            head_kwargs.update(task_head_overrides)

        self.task_head = get_task_head(
            task["task_type"], 
            decoder_type=task["decoder_type"], 
            num_classes=task["num_classes"],
            input_size=foundation_model["embed_dim"],
            **head_kwargs
            )
        self.task_info = task
        self.train_metrics = None
        self.val_metrics = None
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

    @classmethod
    def from_config(cls, config: dict):
        """Instantiate the adapter directly from a config dictionary."""
        if "foundation_model" not in config:
            raise ValueError("Config missing 'foundation_model' section")
        if "task" not in config:
            raise ValueError("Config missing 'task' section")

        training_cfg = config.get("training", {})
        task_cfg = deepcopy(config["task"])
        task_head_overrides = training_cfg.get("task_head_overrides")
        dropout_override = training_cfg.get("dropout")
        if dropout_override is not None:
            task_head_overrides = {**(task_head_overrides or {}), "dropout_rate": dropout_override}
        return cls(
            foundation_model=config["foundation_model"],
            task=task_cfg,
            freeze_backbone=training_cfg.get("freeze_backbone", True),
            learning_rate=training_cfg.get("learning_rate", 1e-3),
            optimizer_config=training_cfg.get("optimizer"),
            scheduler_config=training_cfg.get("scheduler"),
            label_smoothing=training_cfg.get("label_smoothing", 0.0),
            task_head_overrides=task_head_overrides,
        )

    def forward(self, x, y, *args):
        if self.freeze_backbone:
            with torch.no_grad():
                representations = self.backbone(x)
        else:
            representations = self.backbone(x)
        representations = representations.flatten(start_dim=1)

        logits = self.task_head(representations)

        if self.task_info["task_type"] == "classification":
            loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
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
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def _build_optimizer(self):
        config = dict(self.optimizer_config)
        opt_name = config.pop("name", "AdamW").lower()
        optimizer_map = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "stableadamw": torch.optim.AdamW,
        }
        optimizer_cls = optimizer_map.get(opt_name)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer '{opt_name}'")
        if "betas" in config:
            config["betas"] = tuple(config["betas"])
        if "weight_decay" not in config:
            config["weight_decay"] = 0.0
        return optimizer_cls(self.parameters(), lr=self.learning_rate, **config)

    def _build_scheduler(self, optimizer):
        if not self.scheduler_config:
            return None
        config = dict(self.scheduler_config)
        scheduler_name = config.pop("name", None)
        if not scheduler_name:
            return None
        interval = config.pop("interval", "epoch")
        monitor = config.pop("monitor", "val_loss")
        frequency = config.pop("frequency", 1)
        warmup_steps = config.pop("warmup_steps", 0)
        warmup_start_factor = config.pop("warmup_start_factor", 0.1)

        scheduler_cls = getattr(torch_lr_scheduler, scheduler_name, None)
        if scheduler_cls is None:
            raise ValueError(f"Unsupported scheduler '{scheduler_name}'")
        scheduler = scheduler_cls(optimizer, **config)

        if warmup_steps:
            warmup = torch_lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                total_iters=warmup_steps,
            )
            scheduler = torch_lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
                milestones=[warmup_steps],
            )

        scheduler_dict = {
            "scheduler": scheduler,
            "interval": interval,
            "monitor": monitor,
        }
        if frequency is not None:
            scheduler_dict["frequency"] = frequency
        return scheduler_dict
