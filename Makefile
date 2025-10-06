# Makefile for kbench-controls

.PHONY: help install install-dev clean test docs format lint

help:
	@echo "Available commands:"
	@echo "  install     - Install production environment"
	@echo "  clean       - Clean temporary files"
	@echo "  test        - Run tests"
	@echo "  docs        - Build documentation"
	@echo "  format      - Format code with black"
	@echo "  lint        - Check code with flake8"

install:
	conda env create -f environment.yml
	@echo "Environment installed. To activate it, use: conda activate Kbench"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/

test:
	pytest

docs:
	cd docs && make html

format:
	black src/ scripts/

lint:
	flake8 src/ scripts/

update-env:
	conda env update -f environment.yml

update-env-dev:
	conda env update -f environment-dev.yml