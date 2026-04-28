import os
import sys
import pandas as pd
import joblib
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import PROCESSED_DATA_DIR, MODELS_DIR, BEST_LSTM_CONFIG, ASSET_CONFIG
from src.data.preprocessing import create_sequences

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def get_prefix(asset):
    return 'btc' if asset == 'Bitcoin' else asset.lower()

def run_backup_inference():
    report_lines = []
    report_lines.append("# 🛡️ DeepForecast AI - Static Backup Report")
    report_lines.append(f"**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n*This document serves as a static fallback in the event of internet failure, API rate-limiting, or local server crashes during the live demonstration.*")
    
    for asset in ["Bitcoin", "Gold", "Silver"]:
        report_lines.append(f"\n## 📈 Asset: {asset}")
        prefix = get_prefix(asset)
        
        # Load Model
        model_path = os.path.join(MODELS_DIR, f'{prefix}_lstm_final.keras')
        scaler_path = os.path.join(MODELS_DIR, f'{prefix}_scaler.pkl')
        
        if not os.path.exists(model_path):
            report_lines.append("> ❌ Model not found.")
            continue
            
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load Data
        live_path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_live_scaled.csv')
        hist_path = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_features.csv')
        
        if os.path.exists(live_path):
            df_scaled = pd.read_csv(live_path, index_col='timestamp', parse_dates=True)
            df_raw = pd.read_csv(hist_path, index_col='timestamp', parse_dates=True)
            
            latest_price = df_raw['price'].iloc[-1]
            latest_date = df_raw.index[-1].strftime('%Y-%m-%d')
            
            seq_len = BEST_LSTM_CONFIG['seq_len']
            n_features = df_scaled.shape[1]
            price_col_idx = list(df_scaled.columns).index('price')
            
            last_sequence = df_scaled.values[-seq_len:].reshape(1, seq_len, n_features)
            
            # Predict
            tomorrow_scaled = model.predict(last_sequence, verbose=0)
            
            import numpy as np
            dummy = np.zeros((1, n_features))
            dummy[0, price_col_idx] = tomorrow_scaled[0][0]
            tomorrow_usd = scaler.inverse_transform(dummy)[0, price_col_idx]
            
            diff = tomorrow_usd - latest_price
            pct = (diff / latest_price) * 100
            
            indicator = "🟢 UPTREND" if diff >= 0 else "🔴 DOWNTREND"
            
            report_lines.append(f"- **Latest Known Close ({latest_date}):** ${latest_price:,.2f}")
            report_lines.append(f"- **AI Target (T+1):** ${tomorrow_usd:,.2f}")
            report_lines.append(f"- **Predicted Move:** {diff:+,.2f} ({pct:+.2f}%) {indicator}")
            
        else:
            report_lines.append("> ❌ Scaled data not found.")

    report_path = os.path.join(RESULTS_DIR, 'static_demo_backup.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"Static backup report generated at: {report_path}")

if __name__ == "__main__":
    run_backup_inference()
