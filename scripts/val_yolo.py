"""Validate a trained YOLO model.

Usage:
    python scripts/val_yolo.py \
        --weights runs/my_experiment/weights/best.pt \
        --dataset-yaml data/my_dataset.yaml \
        --val-cfg scripts/configs/val.yaml \
        --save-dir runs/my_experiment/val
"""

import sys
import argparse
from pathlib import Path

# Ensure the local repo is imported instead of any pip-installed ultralytics.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a YOLO model (Ultralytics wrapper)")
    parser.add_argument("--weights", required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--dataset-yaml", required=True, help="Path to dataset YAML")
    parser.add_argument("--val-cfg", required=True, help="Path to validation config YAML")
    parser.add_argument("--save-dir", default=None, help="Root directory to save validation results")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.val_cfg) as f:
        cfg = yaml.safe_load(f) or {}

    cfg["data"] = str(Path(args.dataset_yaml).resolve())

    if args.save_dir:
        save_dir = Path(args.save_dir)
        cfg["project"] = str(save_dir.parent.resolve())
        cfg["name"] = save_dir.name

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(**cfg)

    print(f"\nValidation complete.")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP75    : {metrics.box.map75:.4f}")


if __name__ == "__main__":
    main()
