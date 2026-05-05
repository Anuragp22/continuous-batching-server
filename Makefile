.PHONY: install dev test test-fast lint format typecheck bench-naive bench-batched bench clean docker-up docker-down

PY ?= python
UV ?= uv
MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
CONCURRENCY ?= 8
HOST ?= 0.0.0.0
PORT ?= 8000

install:
	$(UV) sync --extra dev --extra bench

install-gpu:
	$(UV) sync --extra dev --extra bench --extra gpu

dev:
	$(UV) run uvicorn cbserver.api.app:create_app --factory --host $(HOST) --port $(PORT) --reload

serve:
	$(UV) run cbserver serve --model $(MODEL) --host $(HOST) --port $(PORT)

test:
	$(UV) run pytest -v

test-fast:
	$(UV) run pytest -v -m "not slow"

lint:
	$(UV) run ruff check src tests bench

format:
	$(UV) run ruff format src tests bench
	$(UV) run ruff check --fix src tests bench

typecheck:
	$(UV) run mypy src

bench-naive:
	$(UV) run python -m bench.harness --mode naive --model $(MODEL) --concurrency $(CONCURRENCY) --output bench/results/m1_naive.json

bench-batched:
	$(UV) run python -m bench.harness --mode batched --model $(MODEL) --concurrency $(CONCURRENCY) --output bench/results/m3_batched.json

bench:
	$(UV) run python -m bench.harness --mode sweep --model $(MODEL) --output bench/results/sweep.json

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
