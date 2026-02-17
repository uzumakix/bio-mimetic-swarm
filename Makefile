.PHONY: run test lint clean

run:
	python -m src

test:
	python -m pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
