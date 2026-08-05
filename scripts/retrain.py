#!/usr/bin/env python3
import os
import sys
import logging
import subprocess

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'retrain.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    logger.info(f"Starting: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        logger.info(f"Completed: {description}\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}\nError: {e.stderr}")
        sys.exit(1)

def main():
    logger.info("Starting automated retraining pipeline...")
    
    # 1. Refresh Data
    run_command("python src/data/data_collection.py", "Data Collection")
    run_command("python src/data/external_data.py", "External Data Collection")
    
    # 2. Preprocess Data
    run_command("python src/data/preprocessing.py", "Data Preprocessing")
    
    # 3. Train Models
    run_command("python src/training/train_lgbm.py", "LightGBM Training")
    run_command("python src/training/train_catboost.py", "CatBoost Training")
    run_command("python src/training/train_deep_learning.py", "Deep Learning Training")
    
    # 4. Evaluate Models (assuming this is done during training or a separate script)
    # If there's an evaluation script, we could run it here.
    logger.info("Retraining pipeline completed successfully.")
    
if __name__ == "__main__":
    main()
