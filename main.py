import argparse
import yaml
from lightning import Trainer

from eeg_finetuner.finetune import FEMTaskAdapter
from eeg_finetuner.data import get_dataloaders 

def main():
    parser = argparse.ArgumentParser(description="Run EEG finetuning experiment")
    parser.add_argument("--config", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration if no config file is provided
        config = {
            "foundation_model": {
                "model_name": "labram",
                "input_size": 128,
                "num_channels": 64,
                "embedding_size": 128
            },
            "task": {
                "task_type": "classification",
                "num_classes": 2,
                "decoder_type": "linear",
            },
            "training": {
                "freeze_backbone": True,
                "learning_rate": 1e-3,
                "max_epochs": 10
            },
            "data": {
                "dataset": "example_eeg_dataset",
                "batch_size": 32,
            }
        }

    finetuner = FEMTaskAdapter(
        foundation_model=config["foundation_model"],
        task=config["task"],
        freeze_backbone=config.get("training", {}).get("freeze_backbone", True),
        learning_rate=config.get("training", {}).get("learning_rate", 1e-3)
    )
    
    train_dataloader, val_dataloader = get_dataloaders(config["data"])
    trainer = Trainer(max_epochs=config.get("training", {}).get("max_epochs", 10))
    trainer.fit(finetuner, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()