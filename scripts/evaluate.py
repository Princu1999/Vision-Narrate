#!/usr/bin/env python3
"""Evaluation entrypoint (uses src/visionnarrate/metrics.py)."""
import argparse
from pathlib import Path

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', type=Path, required=True, help='Predictions file (json/csv)')
    p.add_argument('--gold', type=Path, required=True, help='Ground truth file (json/csv)')
    p.add_argument('--out', type=Path, default=Path('./metrics.json'))
    return p.parse_args()

def main():
    args = get_args()
    try:
        from src.visionnarrate import metrics
    except Exception as e:
        print("Install requirements first. Error:", e); return

    if hasattr(metrics, 'evaluate_cli'):
        metrics.evaluate_cli(args)
    else:
        print("Add an evaluate_cli(args) function in src/visionnarrate/metrics.py to enable CLI usage.")

if __name__ == '__main__':
    main()
