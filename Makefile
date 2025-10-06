# Makefile pour kbench-controls

.PHONY: help install install-dev clean test docs format lint

help:
	@echo "Commandes disponibles:"
	@echo "  install     - Installer l'environnement de production"
	@echo "  clean       - Nettoyer les fichiers temporaires"
	@echo "  test        - Lancer les tests"
	@echo "  docs        - Construire la documentation"
	@echo "  format      - Formater le code avec black"
	@echo "  lint        - Vérifier le code avec flake8"

install:
	conda env create -f environment.yml
	@echo "Environnement installé. Pour l'activer, utilisez : conda activate Kbench"

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