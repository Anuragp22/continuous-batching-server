# cbserver — Continuous-Batching Inference Server

A from-scratch mini-vLLM clone: continuous-batching scheduler, paged KV cache, streaming SSE API.

## Status

Work in progress. See [`plan.md`](plan.md) for the four-milestone roadmap.

| Milestone | Scope | State |
|---|---|---|
| M1 | FastAPI skeleton + SSE + naive `model.generate()` baseline + Prometheus/Grafana | in progress |
| M2 | Manual prefill/decode loop with `DynamicCache`, parity vs HF `generate()` | not started |
| M3 | Continuous-batching scheduler with token-budget admission | not started |
| M4 | Paged KV cache + benchmark sweep + README plots | not started |

## Quick start

```bash
uv sync --extra dev
make dev      # uvicorn at http://localhost:8000
make test     # unit + integration
```

## Layout

```
src/cbserver/
├── api/         # FastAPI app, routes, SSE
├── engine/      # AsyncEngine outer loop
├── scheduler/   # FCFS + token-budget admission
├── cache/       # paged KV pool + block manager
├── model/       # executor, attention, sampling
└── obs/         # prometheus + structlog
bench/           # workloads, harness, results
tests/           # unit / integration / e2e
```

## Targets (final acceptance)

1. ≥5× decode-throughput vs naive sequential serving on Zipfian workload (A100)
2. p99 TTFT < 500 ms at concurrency 32
3. Zero KV-cache leaks under 10 000-request soak
4. Cancellation reflected in active set within one scheduler tick (≤50 ms)

Every numeric claim must be reproducible from `bench/results/` with commit SHA + hardware stamped into the JSON.
