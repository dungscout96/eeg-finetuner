"""Experiment orchestration utilities."""

from __future__ import annotations

from lightning import Trainer

from .data import DatasetPipeline
from .finetune import FEMTaskAdapter
from .generate_model_card import generate_model_card


def build_dataloaders(data_config: dict):
    """Create train/val/test dataloaders based on the provided data config."""
    if not data_config:
        raise ValueError("Data configuration is required to build dataloaders")
    if "dataset" not in data_config:
        raise ValueError("Data configuration must include a 'dataset' key")

    dataset_name = data_config["dataset"]
    overrides = {key: value for key, value in data_config.items() if key != "dataset"}
    pipeline = DatasetPipeline(dataset_name, overrides=overrides)
    return pipeline.process_dataset()


def prepare_trainer(training_cfg: dict | None = None) -> Trainer:
    training_cfg = training_cfg or {}
    trainer_kwargs = {"max_epochs": training_cfg.get("max_epochs", 10)}
    gradient_clip_val = training_cfg.get("gradient_clip_val")
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_val
    return Trainer(**trainer_kwargs)


def run_experiment(config: dict):
    """Run finetuning experiment and return a model card."""
    finetuner = FEMTaskAdapter.from_config(config)
    train_loader, val_loader, test_loader = build_dataloaders(config.get("data", {}))
    trainer = prepare_trainer(config.get("training"))
    trainer.fit(finetuner, train_loader, val_loader)

    evaluation_loader = test_loader or train_loader
    model_card = generate_model_card(
        config=config,
        finetuned_model=finetuner,
        test_dataloader=evaluation_loader,
    )
    return model_card
