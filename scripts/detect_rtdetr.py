"""Run inference with a trained RT-DETR model.

Usage:
    python scripts/detect_rtdetr.py \
        --weights runs/my_rtdetr_experiment/weights/best.pt \
        --source path/to/images_or_video \
        --detect-cfg scripts/configs/detect.yaml \
        --save-dir runs/my_rtdetr_experiment/inference
"""

import sys
import argparse
from pathlib import Path

# Ensure the local repo is imported instead of any pip-installed ultralytics.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run RT-DETR inference (Ultralytics wrapper)")
    parser.add_argument("--weights", required=True, help="Path to trained model weights (.pt) or exported model (.onnx, etc.)")
    parser.add_argument("--source", required=True, help="Input source: image/video path, directory, URL, or 0 for webcam")
    parser.add_argument("--detect-cfg", required=True, help="Path to detection/inference config YAML")
    parser.add_argument("--save-dir", default=None, help="Root directory to save detection results")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.detect_cfg) as f:
        cfg = yaml.safe_load(f) or {}

    cfg["source"] = args.source

    if args.save_dir:
        save_dir = Path(args.save_dir)
        cfg["project"] = str(save_dir.parent.resolve())
        cfg["name"] = save_dir.name

    from ultralytics import RTDETR

    model = RTDETR(args.weights)
    results = model.predict(**cfg)

    n = len(results)
    print(f"\nInference complete. Processed {n} image(s).")
    if args.save_dir or cfg.get("save", True):
        print(f"Results saved to: {Path(cfg.get('project', 'runs')) / cfg.get('name', 'predict')}")


if __name__ == "__main__":
    main()
