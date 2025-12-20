def get_dataloaders(data_config):
    # Placeholder function to return dataloaders based on data_config
    # In practice, this would load datasets, apply transforms, and create DataLoader objects
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    # Example dummy data
    X_train = torch.randn(100, 128)  # 100 samples, 64 channels, 128 time points
    y_train = torch.randint(0, 2, (100,))  # Binary classification

    X_val = torch.randn(20, 128)  # 20 samples, 64 channels, 128 time points
    y_val = torch.randint(0, 2, (20,))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=data_config.get("batch_size", 32), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=data_config.get("batch_size", 32), shuffle=False)

    return train_dataloader, val_dataloader