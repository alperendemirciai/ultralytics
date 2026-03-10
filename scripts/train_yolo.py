"""Train a YOLO model.

Usage:
    python scripts/train_yolo.py \
        --dataset-yaml data/my_dataset.yaml \
        --train-cfg scripts/configs/train.yaml \
        --model yolo11n.yaml \
        --save-dir runs/my_experiment
"""

import sys
import argparse
from pathlib import Path

# Ensure the local repo is imported instead of any pip-installed ultralytics.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model (Ultralytics wrapper)")
    parser.add_argument("--dataset-yaml", required=True, help="Path to dataset YAML (defines train/val paths, nc, names, optionally bit_depth/channels)")
    parser.add_argument("--train-cfg", required=True, help="Path to training config YAML (hyperparameters)")
    parser.add_argument("--model", default="yolo11n.yaml", help="Model architecture YAML or pretrained .pt weights (default: yolo11n.yaml)")
    parser.add_argument("--save-dir", default=None, help="Root directory to save training results")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.train_cfg) as f:
        cfg = yaml.safe_load(f) or {}

    # dataset yaml and model are passed explicitly; save-dir maps to project/name
    cfg["data"] = str(Path(args.dataset_yaml).resolve())

    if args.save_dir:
        save_dir = Path(args.save_dir)
        cfg["project"] = str(save_dir.parent.resolve())
        cfg["name"] = save_dir.name

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(**cfg)
    print(f"\nTraining complete. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
