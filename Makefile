all: run

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb src/main.py

clean:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache

lint:
	flake8 .
	mypy . --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

.PHONY: all install run debug clean lint