.PHONY: help install install-dev test test-cov clean build upload-test upload format lint

help:
	@echo "Available commands:"
	@echo "  install      Install package"
	@echo "  install-dev  Install package in development mode with dev dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run linting with flake8"

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=pyoptv --cov-report=html --cov-report=term

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload-test: build
	python -m twine upload --repository testpypi dist/*

upload: build
	python -m twine upload dist/*

format:
	black pyoptv tests
	isort pyoptv tests

lint:
	flake8 pyoptv tests
