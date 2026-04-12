import pandas as pd
import os

# Test directory structure from the perspective of the notebooks/ folder
PROCESSED_DIR = 'data/processed/' # From root
# or 
PROCESSED_DIR_REL = './data/processed/'

print("Checking processed data files...")
assets = ['bitcoin_data.csv', 'gold_data.csv', 'silver_data.csv']
for f in assets:
    path = os.path.join('data/processed/', f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✅ FOUND: {f} ({len(df)} rows)")
    else:
        print(f"❌ MISSING: {path}")
