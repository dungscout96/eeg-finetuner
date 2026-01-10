import argparse
import json
import yaml

from eeg_finetuner.experiment import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run EEG finetuning experiment")
    parser.add_argument("--config", type=str, help="Path to the config.yaml file", default="/Users/dtyoung/Documents/Research/LEM-SCCN/standardized-finetuning/config.yaml")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    model_card = run_experiment(config)
    print(json.dumps(model_card, indent=2))  # This can be logged to a logging service

if __name__ == "__main__":
    main()