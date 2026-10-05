.PHONY: install test lint run-api run-etl run-train docker-up docker-down clean

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

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
