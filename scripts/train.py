#!/usr/bin/env python3
"""Train entrypoint (calls your training loop from src/visionnarrate/train_utils.py). Edit as needed."""
import argparse
from pathlib import Path

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=False, help='Path to training dataset')
    p.add_argument('--output', type=Path, default=Path('./outputs'))
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = get_args()
    # Lazy imports to keep CLI snappy
    try:
        import torch
        from src.visionnarrate import train_utils, models, datasets
    except Exception as e:
        print("Install requirements first. Error:", e); return

    if hasattr(train_utils, 'train'):
        train_utils.train(args)
    elif hasattr(train_utils, 'train_one_run'):
        train_utils.train_one_run(args)
    else:
        print("No train() or train_one_run() found in train_utils.py. Please wire up your training loop.")

if __name__ == '__main__':
    main()
