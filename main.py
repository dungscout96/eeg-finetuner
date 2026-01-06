import argparse
import yaml
from lightning import Trainer

from eeg_finetuner.finetune import FEMTaskAdapter
from eeg_finetuner.data import DatasetPipeline
from eeg_finetuner.generate_model_card import generate_model_card, pretty_print_model_card

def main():
    parser = argparse.ArgumentParser(description="Run EEG finetuning experiment")
    parser.add_argument("--config", type=str, help="Path to the config.yaml file", default="/Users/dtyoung/Documents/Research/LEM-SCCN/standardized-finetuning/config.yaml")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    finetuner = FEMTaskAdapter(
        foundation_model=config["foundation_model"],
        task=config["task"],
        freeze_backbone=config.get("training", {}).get("freeze_backbone", True),
        learning_rate=config.get("training", {}).get("learning_rate", 1e-3)
    )
    
    data_pipeline = DatasetPipeline(config["data"]["dataset"])
    train_dataloader, val_dataloader, test_dataloader = data_pipeline.process_dataset()
    trainer = Trainer(max_epochs=config.get("training", {}).get("max_epochs", 10))
    trainer.fit(finetuner, train_dataloader, val_dataloader)

    model_card = generate_model_card(
        config=config,
        finetuned_model=finetuner,
        test_dataloader=train_dataloader, # TODO switch to test_dataloader
    )
    pretty_print_model_card(model_card)

if __name__ == "__main__":
    main()