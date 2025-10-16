#!/usr/bin/env python3
"""Inference entrypoint (uses src/visionnarrate/inference.py)."""
import argparse
from pathlib import Path

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=False)
    p.add_argument('--image', type=Path, required=False, help='Single image for captioning')
    p.add_argument('--images_dir', type=Path, required=False, help='Directory of images')
    p.add_argument('--out', type=Path, default=Path('./predictions.json'))
    return p.parse_args()

def main():
    args = get_args()
    try:
        from src.visionnarrate import inference
    except Exception as e:
        print("Install requirements first. Error:", e); return

    if hasattr(inference, 'run_cli'):
        inference.run_cli(args)
    else:
        print("Add a run_cli(args) function in src/visionnarrate/inference.py to enable CLI usage.")

if __name__ == '__main__':
    main()
