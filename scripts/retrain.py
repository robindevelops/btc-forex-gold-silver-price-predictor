#!/usr/bin/env python3
"""
Automated end-to-end retraining: data → preprocessing → tuning → training → stacking → evaluation → figures.

    python scripts/retrain.py            # full run on the frozen data/raw snapshot (tuning included, slow)
    python scripts/retrain.py --no-tune  # reuse results/tuning/best_params.json
    python scripts/retrain.py --fetch    # FIRST re-download data/raw from Yahoo — this REPLACES the frozen snapshot behind
                                         # every reported result (Yahoo revises history and rolls futures contracts)

The default never downloads, so a plain run cannot silently overwrite the frozen evaluation data.
"""
import os
import sys
import logging
import argparse
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = os.path.join(ROOT, 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(log_dir, 'retrain.log')), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def run(cmd, description):
    logger.info(f"Starting: {description}")
    env = {**os.environ, 'PYTHONPATH': ROOT, 'TF_CPP_MIN_LOG_LEVEL': '3'}
    result = subprocess.run([sys.executable] + cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error(f"Failed: {description}\n{result.stderr[-3000:]}")
        sys.exit(1)
    logger.info(f"Completed: {description}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true',
                    help='re-download data/raw first (replaces the frozen snapshot behind the reported results)')
    ap.add_argument('--no-fetch', action='store_true', help='kept for backward compatibility; not fetching is now the default')
    ap.add_argument('--no-tune', action='store_true')
    args = ap.parse_args()
    if args.fetch and not args.no_fetch:
        run(['src/data/data_collection.py'], 'Data collection')
        run(['src/data/external_data.py'], 'External data collection')
    run(['src/data/preprocessing.py'], 'Preprocessing')
    run(['src/data/eda.py'], 'EDA')
    if not args.no_tune:
        run(['src/training/tune_models.py'], 'Walk-forward tuning')
    run(['src/training/train_models.py'], 'Training')
    run(['src/models/ensemble_model.py'], 'Stacking')
    run(['src/evaluation/backtesting.py'], 'Final evaluation')
    run(['src/evaluation/plots.py'], 'Figures')
    run(['src/experiments/run_experiments.py'], 'Design experiments')
    logger.info("Retraining pipeline completed successfully.")


if __name__ == "__main__":
    main()
