all: run

install:
	uv sync

run:
	uv run python -m src.__main__

clean:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache

lint:
	flake8 src/.

.PHONY: run clean lint