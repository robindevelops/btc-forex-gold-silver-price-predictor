.PHONY: install test collect-data preprocess eda tune train stack evaluate figures experiments report pipeline serve api docker-build docker-run clean

PY ?= python

install:
	pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

collect-data:            ## download OHLCV + macro + sentiment (needs network; Yahoo may rate-limit)
	$(PY) src/data/data_collection.py
	$(PY) src/data/external_data.py

preprocess:              ## clean, engineer features, frozen chronological split, fit scaler on train
	$(PY) src/data/preprocessing.py

eda:                     ## ADF tests, ACF/PACF, return distributions
	$(PY) src/data/eda.py

tune:                    ## walk-forward hyper-parameter search on train+val only (~30 min, GRU/LSTM dominate)
	$(PY) src/training/tune_models.py

train:                   ## two-phase training of every model with the tuned parameters
	$(PY) src/training/train_models.py

stack:                   ## stacked-ensemble experiment (OOF meta-model)
	$(PY) src/models/ensemble_model.py

evaluate:                ## ONE evaluation on the untouched test set + model selection by CV
	$(PY) src/evaluation/backtesting.py

figures:                 ## report figures into results/figures/
	$(PY) src/evaluation/plots.py

experiments:             ## design experiments (validation only) + before/after comparison on identical unseen days
	$(PY) src/experiments/run_experiments.py
	$(PY) src/experiments/before_after.py

report:                  ## regenerate docs/FYP_Final_Report.pdf and docs/Executive_Summary.pdf
	$(PY) docs/build_report.py

pipeline: preprocess eda tune train stack evaluate figures experiments test   ## full reproducible run from raw data

predict:
	$(PY) src/inference/prediction.py

serve:
	streamlit run app/streamlit_app.py

api:
	uvicorn src.api.app:app --reload

docker-build:
	docker build -t crypto-forex-predictor .

docker-run:
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ catboost_info/
