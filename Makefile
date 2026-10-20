.PHONY: install test lint run-api run-etl run-train prefect-setup prefect-deploy prefect-worker docker-up docker-down clean

install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -v

lint:
	ruff check src tests

run-api:
	uvicorn fleetsense.api.main:app --host 0.0.0.0 --port 8000 --reload

run-etl:
	python -m fleetsense.workflows.etl

run-train:
	python -m fleetsense.workflows.train

mlflow-ui:
	mlflow ui --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflow --port 5000

prefect-setup:
	prefect work-pool create fleetsense-pool --type process || true

prefect-deploy:
	prefect deploy --all

prefect-worker:
	prefect worker start --pool fleetsense-pool

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
