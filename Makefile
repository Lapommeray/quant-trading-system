.PHONY: install test lint format clean docker-build docker-run help

PYTHON := python3
PIP := pip
VENV := venv
DOCKER_IMAGE := quant-trading-system
DOCKER_TAG := latest

help:
	@echo "Available commands:"
	@echo "  install       - Install dependencies and setup environment"
	@echo "  test          - Run test suite"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code with black and isort"
	@echo "  clean         - Clean up temporary files"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  setup-dev     - Setup development environment"

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install -r requirements-dev.txt
	$(VENV)/bin/pip install -e .

setup-dev: install
	$(VENV)/bin/pre-commit install
	cp .env.example .env
	mkdir -p logs

test:
	$(VENV)/bin/pytest tests/ -v
	$(VENV)/bin/python comprehensive_testing_framework.py
	$(VENV)/bin/python run_complete_enhanced_test.py

lint:
	$(VENV)/bin/flake8 core modules advanced_modules
	$(VENV)/bin/pylint core modules --fail-under=8.0
	$(VENV)/bin/mypy core modules

format:
	$(VENV)/bin/black .
	$(VENV)/bin/isort .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

verify-install:
	$(VENV)/bin/python -c "import core, modules; print('✅ Core modules imported successfully')"
	$(VENV)/bin/python -c "import pytest; print('✅ Testing framework available')"
	$(VENV)/bin/python -c "import yaml; print('✅ Configuration support available')"

all: clean install test lint

dev: setup-dev verify-install
	@echo "🚀 Development environment ready!"
	@echo "Activate with: source $(VENV)/bin/activate"
