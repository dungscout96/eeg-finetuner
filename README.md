# EEG Finetuner

A standardized framework for fine-tuning EEG foundation models using PyTorch Lightning. This package provides a unified interface to adapt various EEG backbones with different classification or regression heads for downstream tasks.

## Features

- **Standardized Fine-tuning**: Uses PyTorch Lightning (`EEGFineTuner`) for a consistent training workflow.
- **Modular Backbones**: Support for EEG foundation models like LaBRAM.
- **Flexible Task Heads**: Easily switch between Linear and MLP heads for classification or regression.
- **Configuration-Driven**: Define models, tasks, and data using simple dictionaries.
- **Integration**: Built on top of `braindecode`, `eegdash`, and `lightning`.

## Installation

This project uses `uv` for dependency management.

```bash
# Clone the repository
git clone <repository-url>
cd eeg-finetuner

# Install dependencies
uv sync
```

Alternatively, using `pip`:

```bash
pip install .
```

## Project Structure

```text
eeg_finetuner/
├── foundation_model/    # EEG backbone implementations
│   └── labram/          # LaBRAM model support
├── task_head.py         # Task head definitions (Linear, MLP)
├── finetune.py          # PyTorch Lightning adapter (EEGFineTuner)
├── data.py              # Data loading utilities
└── main.py              # Example entry point
```

## Usage

### Running with a Configuration File

You can run experiments using a YAML configuration file. This is the recommended way to manage different experimental setups.

```bash
python eeg_finetuner/main.py --config config.yaml
```

Example `config.yaml`:

```yaml
foundation_model:
  model_name: "labram"
  input_size: 128
  num_channels: 64
  embedding_size: 128
task:
  task_type: "classification"
  num_classes: 2
  decoder_type: "linear"
training:
  freeze_backbone: true
  learning_rate: 0.001
  max_epochs: 10
data:
  dataset: "example_eeg_dataset"
  batch_size: 32
```

### Basic Example (Python API)

```python
import lightning as L
from eeg_finetuner.finetune import FEMTaskAdapter
from eeg_finetuner.data import get_dataloaders

# 1. Define configurations
foundation_model_config = {
    "model_name": "labram",
    "input_size": 128,
    "num_channels": 64,
    "embedding_size": 128
}

task_config = {
    "task_type": "classification", # or "regression"
    "num_classes": 2,
    "decoder_type": "linear",      # or "mlp"
}

# 2. Initialize the finetuner
model = FEMTaskAdapter(
    foundation_model=foundation_model_config,
    task=task_config,
    freeze_backbone=True,
    learning_rate=1e-3
)

# 3. Get dataloaders
data_config = {"batch_size": 32}
train_loader, val_loader = get_dataloaders(data_config)

# 4. Train with PyTorch Lightning
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)
```

## Components

### Foundation Models
Currently supported:
- `labram`: Large-scale Brain Model (LaBRAM) backbone.
    - `input_size`: Length of the input EEG segments.
    - `num_channels`: Number of EEG channels.
    - `embedding_size`: Dimension of the output representation from the backbone.

### Task Heads
- `linear`: A simple linear layer with dropout.
- `mlp`: A multi-layer perceptron with configurable hidden sizes and dropout.

#### Task Configuration
- `task_type`: `"classification"` or `"regression"`.
- `decoder_type`: `"linear"` or `"mlp"`.
- `num_classes`: Number of output classes (for classification) or 1 (for regression).

## License

MIT License (or specify your license here)
