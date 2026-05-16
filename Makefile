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

# lint:
# 	cd src/ && flake8 --exclude=.venv .
# 	cd src/ && mypy . \
# 		--warn-return-any \
# 		--warn-unused-ignores \
# 		--ignore-missing-imports \
# 		--disallow-untyped-defs \
# 		--check-untyped-
		
lint:
	cd src/ && flake8 --exclude=.venv .
	cd src/ && mypy . --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--follow-imports=skip

.PHONY: all install run debug clean lint