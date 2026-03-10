"""Train an RT-DETR model.

Usage:
    python scripts/train_rtdetr.py \
        --dataset-yaml data/my_dataset.yaml \
        --train-cfg scripts/configs/train.yaml \
        --model rtdetr-l.yaml \
        --save-dir runs/my_rtdetr_experiment

    python scripts/train_rtdetr.py \
        --dataset-yaml /Users/alperendemirci/Documents/GitHub/ultralytics/IRSTD_DATASETS/IRSTD-1K-YOLO_1ch_14bit/dataset.yaml \
        --train-cfg /Users/alperendemirci/Documents/GitHub/ultralytics/scripts/configs/train.yaml \
        --model rtdetr-l.yaml \
        --save-dir runs/my_rtdetr_experiment
"""

import sys
import argparse
from pathlib import Path

# Ensure the local repo is imported instead of any pip-installed ultralytics.
# This is required because this repo contains custom extensions (bit_depth,
# augment_train, etc.) that are not present in the published package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RT-DETR model (Ultralytics wrapper)")
    parser.add_argument("--dataset-yaml", required=True, help="Path to dataset YAML (defines train/val paths, nc, names, optionally bit_depth/channels)")
    parser.add_argument("--train-cfg", required=True, help="Path to training config YAML (hyperparameters)")
    parser.add_argument("--model", default="rtdetr-l.yaml", help="Model architecture YAML or pretrained .pt weights (default: rtdetr-l.yaml)")
    parser.add_argument("--save-dir", default=None, help="Root directory to save training results")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.train_cfg) as f:
        cfg = yaml.safe_load(f) or {}

    cfg["data"] = str(Path(args.dataset_yaml).resolve())

    if args.save_dir:
        save_dir = Path(args.save_dir)
        cfg["project"] = str(save_dir.parent.resolve())
        cfg["name"] = save_dir.name

    from ultralytics import RTDETR

    model = RTDETR(args.model)
    results = model.train(**cfg)
    print(f"\nTraining complete. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
