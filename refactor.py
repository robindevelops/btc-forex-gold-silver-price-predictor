import os
import glob

base_dir = '/Users/mac/Desktop/crypto-forex-prediction-system/src'

folders = {
    'data': ['data_collection.py', 'preprocessing.py', 'stationarity_tests.py'],
    'models': ['model_lstm.py', 'baseline_models.py', 'arima_model.py', 'ensemble_model.py'],
    'training': ['train_final_btc.py', 'train_final_gold.py', 'train_final_silver.py', 'training.py'],
    'experiments': ['experiment_dropout.py', 'experiment_lstm_units.py', 'experiment_seq_length.py'],
    'evaluation': ['evaluation.py', 'walk_forward.py', 'backtesting.py', 'plot_baselines.py'],
    'inference': ['prediction.py']
}

module_map = {}
for folder, files in folders.items():
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)
    with open(os.path.join(base_dir, folder, '__init__.py'), 'w') as f:
        pass
    for file in files:
        module_name = file.replace('.py', '')
        module_map[f'src.{module_name}'] = f'src.{folder}.{module_name}'

# Rename files
for folder, files in folders.items():
    for file in files:
        old_path = os.path.join(base_dir, file)
        new_path = os.path.join(base_dir, folder, file)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)

# Update imports in all Python files
all_py_files = glob.glob(os.path.join(base_dir, '**', '*.py'), recursive=True)
for py_file in all_py_files:
    with open(py_file, 'r') as f:
        content = f.read()
    
    modified = False
    for old_mod, new_mod in module_map.items():
        if f'from {old_mod} import' in content:
            content = content.replace(f'from {old_mod} import', f'from {new_mod} import')
            modified = True
        if f'import {old_mod}' in content:
            content = content.replace(f'import {old_mod}', f'import {new_mod}')
            modified = True
            
    if modified:
        with open(py_file, 'w') as f:
            f.write(content)
            
print("Refactoring completed successfully!")
