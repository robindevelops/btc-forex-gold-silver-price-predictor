.PHONY: install test lint collect-data preprocess train predict serve docker-build docker-run clean all

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

lint:
	flake8 src/ tests/ || echo "Linting finished"

collect-data:
	python src/data/data_collection.py
	python src/data/external_data.py

preprocess:
	python src/data/preprocessing.py

train:
	python src/training/train_lgbm.py
	python src/training/train_catboost.py
	python src/training/train_deep_learning.py

predict:
	python src/inference/prediction.py

serve:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t crypto-forex-predictor .

docker-run:
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

all: collect-data preprocess train test
