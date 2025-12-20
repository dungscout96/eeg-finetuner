from lightning import Trainer
from finetune import FEMTaskAdapter
from data import get_dataloaders  # Assume this is a module to get your dataloaders
def main():
    # Example configuration
    foundation_model_config = {
        "model_name": "labram",
        "input_size": 128,
        "num_channels": 64,
        "embedding_size": 128
    }
    
    task_config = {
        "task_type": "classification",
        "num_classes": 2,
        "decoder_type": "linear",
    }
    
    finetuner = FEMTaskAdapter(
        foundation_model=foundation_model_config,
        task=task_config,
        freeze_backbone=True,
        learning_rate=1e-3
    )
    
    data_config = {
        "dataset": "example_eeg_dataset",
        "batch_size": 32,
    }
    train_dataloader, val_dataloader = get_dataloaders(data_config)  # Get your dataloaders
    trainer = Trainer(max_epochs=10)
    trainer.fit(finetuner, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()