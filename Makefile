.PHONY: setup lint format test clean

setup:
	python -m venv .venv && .venv/bin/pip install -U pip -r requirements.txt

lint:
	ruff check .

format:
	black . && ruff check --fix .

test:
	pytest -q

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
