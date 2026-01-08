.PHONY: install test lint format notebook clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in dev mode
	pip install -e ".[all]"

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=sim_city --cov-report=term-missing

lint:  ## Run linters
	ruff check src/
	ruff format --check src/

format:  ## Format code
	ruff format src/
	ruff check --fix src/

notebook:  ## Launch Jupyter Lab
	jupyter lab notebooks/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
