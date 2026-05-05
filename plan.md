# Continuous-Batching Inference Server — Implementation Plan

## Context

This plan covers **Project 2** in the user's 2026 CV-build lineup (locked in `c:\Users\anura\Downloads\Devlopment\video-automation\ideas.txt`): a learning-focused mini-vLLM clone that anchors **Backend CV** and **AI Engineer CV** with four defensible bullets:

1. Continuous-batching scheduler with token-budget admission
2. Paged KV-cache pool with LRU eviction
3. Streaming SSE API with sub-tick cancellation
4. Reproducible throughput benchmark vs naive sequential serving

**Why this project:** the rest of the user's portfolio (SynthFlow, JobVoice, SIH RAG) is *AI applications*. None demonstrates **AI infrastructure** — the systems layer that makes hosted LLM inference economically viable. Continuous batching is the dominant 2025-2026 inference pattern (vLLM, SGLang, TGI) and is virtually absent from new-grad CVs. Project gives the user one cross-CV asset with both AI-domain and pure-systems bullets.

**Why now:** 6 weeks part-time (locked), WSL2 dev (locked), apply for Llama 3.2 gating day 1 with ungated fallbacks (locked).

**Bar:** all four bullets must be measurable and defensible — every numeric claim has a reproducible benchmark under `bench/results/` with commit SHA and hardware stamped into the JSON.

## Locked decisions

| | |
|---|---|
| Timeline | 6 weeks part-time, M1-M4 phased (~1 week each, M3+M4 absorb 1 extra week) |
| Dev OS | WSL2 (Ubuntu) on the user's Windows 11 machine |
| Bench hardware | Single A100 40GB on Modal/Runpod, ~$1.50/hr, ~$10-30 total budget |
| Language/stack | Python 3.11 + PyTorch 2.5.1 + transformers 4.46.3 + FastAPI 0.115.5 |
| Models | Qwen 2.5 0.5B Instruct (dev) + SmolLM2 1.7B Instruct (bench) — both ungated |
| Llama 3.2 1B | Apply day 1, opportunistic upgrade — added as a 4th benchmark bar if approval lands by week 5 |

## Repo location + structure

**Path:** `C:\Users\anura\Downloads\Devlopment\video-automation\continuous-batching-server\`

```
continuous-batching-server/
├── pyproject.toml
├── uv.lock
├── README.md                      # CV-facing: arch diagram, benchmark plot, run instructions
├── Makefile                       # make dev, make test, make bench, make lint
├── docker/
│   ├── Dockerfile.cpu             # local dev (WSL2)
│   └── Dockerfile.gpu             # A100 bench (CUDA 12.4 + flash-attn)
├── docker-compose.yml             # server + Prometheus + Grafana
├── prometheus.yml
├── grafana/dashboards/inference.json
├── src/cbserver/
│   ├── config.py                  # pydantic-settings: model, max_seq_len, block_size, max_batch
│   ├── types.py                   # Request, Sequence, Block, BatchInputs (dataclasses)
│   ├── api/
│   │   ├── app.py                 # FastAPI factory
│   │   ├── routes_completions.py  # POST /v1/completions (SSE + non-stream)
│   │   ├── routes_health.py       # /healthz, /metrics
│   │   └── sse.py                 # SSE event formatting + disconnect detection
│   ├── engine/
│   │   ├── engine.py              # AsyncEngine: outer loop = scheduler.step → executor.run → dispatch
│   │   ├── request.py             # SequenceGroup, RequestState (WAITING/RUNNING/FINISHED/CANCELLED)
│   │   └── output.py              # SamplingOutput, StreamEvent
│   ├── scheduler/
│   │   ├── scheduler.py           # FCFS + token-budget admission, prefill/decode interleave
│   │   └── policy.py              # max_num_batched_tokens, max_num_seqs, preemption rules
│   ├── cache/
│   │   ├── block.py               # PhysicalBlock (id, refcount, last_used)
│   │   ├── block_manager.py       # paged allocator + free-list + LRU eviction
│   │   └── kv_pool.py             # tensor-backed KV storage (num_blocks, block_size, layers)
│   ├── model/
│   │   ├── loader.py              # AutoModelForCausalLM with attn_implementation toggle
│   │   ├── executor.py            # ModelExecutor: build batch → forward → sample
│   │   ├── attention_naive.py     # padded batch attention (M3)
│   │   ├── attention_paged.py     # block-table attention gather (M4)
│   │   ├── sampling.py            # greedy + temperature/top-p
│   │   └── hf_baseline.py         # naive sequential reference (M1)
│   ├── obs/
│   │   ├── metrics.py             # prometheus_client counters/histograms/gauges
│   │   └── logging.py             # structlog config
│   └── cli.py                     # cbserver serve, cbserver bench
├── bench/
│   ├── workloads/
│   │   ├── zipfian.py             # Zipf(α=1.2, max=512) prompt distribution
│   │   └── sharegpt_sample.jsonl  # ~500 prompts redistributed
│   ├── harness.py                 # async client driver, computes TTFT/ITL/throughput
│   ├── locustfile.py
│   ├── plots.py                   # matplotlib charts for README
│   └── results/                   # checked-in JSON + PNG
└── tests/
    ├── conftest.py
    ├── unit/                      # block_manager, scheduler, sampling, sse
    ├── integration/               # forward_parity, paged_attention, cancellation, no_kv_leak
    └── e2e/test_api_streaming.py
```

## Dependencies (pinned `pyproject.toml`)

```toml
[project]
name = "cbserver"
requires-python = ">=3.11,<3.12"
dependencies = [
  "torch==2.5.1",
  "transformers==4.46.3",
  "accelerate==1.1.1",
  "tokenizers==0.20.3",
  "safetensors==0.4.5",
  "sentencepiece==0.2.0",
  "fastapi==0.115.5",
  "uvicorn[standard]==0.32.1",
  "sse-starlette==2.1.3",
  "httpx==0.27.2",
  "pydantic==2.9.2",
  "pydantic-settings==2.6.1",
  "structlog==24.4.0",
  "prometheus-client==0.21.0",
  "numpy==1.26.4",
  "click==8.1.7",
]

[project.optional-dependencies]
dev = ["pytest==8.3.3", "pytest-asyncio==0.24.0", "pytest-cov==6.0.0", "pytest-timeout==2.3.1", "ruff==0.7.4", "mypy==1.13.0"]
bench = ["locust==2.32.2", "matplotlib==3.9.2", "pandas==2.2.3", "tqdm==4.67.0"]
gpu = ["flash-attn==2.7.0.post2"]  # WSL2 + A100 only; install with --no-build-isolation
```

The `torch==2.5.1` + `transformers==4.46.3` + `flash-attn==2.7.0.post2` triple is the safe combo as of the May 2026 cutoff — bumping any one breaks the others.

## Phased milestones

### M1 — Skeleton + naive baseline (week 1)

**Goal:** end-to-end SSE works against `model.generate()` with no batching. Establishes the harness for everything else.

**Deliverables:**
- FastAPI app with `POST /v1/completions` (OpenAI-subset: `model`, `prompt`, `max_tokens`, `temperature`, `stream`)
- `model/hf_baseline.py` — one-request-at-a-time `generate(streamer=TextIteratorStreamer)`
- Async semaphore so concurrent requests are queued, not rejected
- `bench/harness.py` v0: spawn N concurrent clients, record TTFT and end-to-end latency
- `bench/workloads/zipfian.py`: Zipf(α=1.2, max=512) prompt-length distribution
- Compose stack: server + Prometheus scraping `/metrics` + Grafana with one starter panel

**Checkpoint:** `make bench-naive MODEL=Qwen/Qwen2.5-0.5B-Instruct CONCURRENCY=8` produces `bench/results/m1_naive.json`. This is the floor every later milestone beats.

**Skills to invoke:**
- `superpowers:writing-plans` — break this milestone into a per-task list at start
- `engineering-skills:senior-backend` — FastAPI + SSE patterns
- `superpowers:test-driven-development` — SSE formatter and request validator (pure logic)
- `engineering-advanced-skills:observability-designer` — Prometheus metric taxonomy (the names you choose here propagate everywhere)
- `superpowers:verification-before-completion` — before declaring M1 done

### M2 — Manual forward pass with per-request KV cache (week 2)

**Goal:** stop calling `generate()`. Replace with our own prefill + token-by-token decode loop using `DynamicCache`. Prove correctness vs HF.

**Educational anchor:** Sebastian Raschka's *LLMs-from-scratch* Chapter 4 `03_kv-cache`. Read his `generate_with_kv_cache` and `generate_no_kv_cache` side-by-side before writing `executor.py`. Our version uses `transformers`' `DynamicCache` instead of a hand-rolled cache, so we get GQA and RoPE handling for free.

**Deliverables:**
- `model/executor.py` with `prefill(input_ids) -> (logits, cache)` and `decode_step(next_token, cache) -> logits`. Uses `past_key_values=DynamicCache()`, `use_cache=True`, `cache_position=...`.
- `engine/engine.py` v1: per-request loop owning one `DynamicCache`; still no batching
- `model/sampling.py`: greedy + temperature + top-p
- `tests/integration/test_forward_parity.py`: 20 prompts × 5 seeds, our output token-for-token == HF `generate()` greedy. **Hard correctness gate.**

**Checkpoint:** parity test green for Qwen 2.5 0.5B. Throughput at concurrency=1 within 20% of `generate()` baseline.

**Skills:**
- `engineering-skills:senior-ml-engineer` — DynamicCache + cache_position details
- `superpowers:test-driven-development` — sampling logic
- `superpowers:systematic-debugging` — when (not if) parity fails. `cache_position` off-by-ones and GQA head-count mismatches are the two likely culprits
- `superpowers:verification-before-completion`

### M3 — Continuous batching + token-budget scheduler (weeks 3-4)

**Goal:** the headline CV bullet. Multiple requests share each forward pass; new arrivals join mid-flight without waiting for the current batch to drain.

**Deliverables:**
- `scheduler/scheduler.py`: WAITING/RUNNING/FINISHED queues. Each `step()` returns `SchedulerOutput(prefills, decodes, preempted)`. Token budget `max_num_batched_tokens` (e.g. 2048) packs prefill chunks + ongoing decodes.
- `scheduler/policy.py`: FCFS for admission, longest-running-first for preemption when memory pressure forces it.
- `model/attention_naive.py`: padded batch attention. Per-request `DynamicCache` instances stitched with `torch.cat` along batch dim each step. Slow but correct — stepping stone before paged.
- `engine/engine.py` v2: `while True: sched_out = scheduler.step(); outputs = executor.run(sched_out); engine.dispatch(outputs)`. Streaming events fan out per-request via `asyncio.Queue`.
- Cancellation v1: client disconnect → request marked `CANCELLED` → scheduler drops it next tick.
- `tests/unit/test_scheduler_admission.py`, `test_scheduler_preemption.py` — pure logic, fully TDD'd with fake clock + fake executor.
- `tests/integration/test_continuous_batching.py`: 32 mixed-length requests; assert all complete and total wall time < naive by ≥3×.

**Checkpoint:** `make bench-batched CONCURRENCY=32` shows ≥3× throughput vs M1 naive on Zipfian workload. (The 5× target is M4 once paged eliminates padding waste.)

**Skills:**
- `engineering-skills:senior-ml-engineer` — batching the forward pass
- `engineering-skills:senior-backend` — asyncio fanout pattern
- `superpowers:test-driven-development` — scheduler is the highest-value TDD target in the entire project
- `engineering-advanced-skills:performance-profiler` — once it works, find the bottleneck (likely Python-side cache concatenation, which motivates M4)
- `superpowers:verification-before-completion`

### M4 — Paged KV cache + cancellation polish + benchmark sweep (weeks 5-6)

**Goal:** replace per-request `DynamicCache` with a shared block pool. Hit all four CV-bullet benchmark targets. Write the README.

**Deliverables:**
- `cache/kv_pool.py`: pre-allocated `(num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim)` tensor. `block_size=16`.
- `cache/block_manager.py`: free-list, refcount per block, LRU eviction → triggers scheduler preemption.
- `cache/block.py` integration: each `Sequence` holds `block_table: list[int]` mapping logical block index → physical block id.
- `model/attention_paged.py`: gather K/V from non-contiguous physical blocks. CPU path uses `index_select`+`einsum` (correct, runs on dev). GPU path tries `flash_attn_with_kvcache` (native block-table support); falls back to gather-based path if flash-attn integration is rocky — correctness over speed.
- `tests/integration/test_paged_attention.py`: paged forward output == padded forward output (atol=1e-4) over 50 random prompt mixes.
- `tests/integration/test_no_kv_leak.py`: 10k-request soak; assert `block_manager.free_blocks()` returns to initial value.
- Cancellation v2: reflected in `active_requests` gauge within one scheduler tick (≤50ms); regression test.
- Bench sweep on rented A100: SmolLM2 1.7B at concurrencies {1, 4, 8, 16, 32, 64}, 3 runs each, vs (a) naive, (b) static-batch (size=8, drains fully before next batch), (c) ours.
- README with architecture diagram (Mermaid), benchmark plots, the four CV bullets phrased exactly.

**Checkpoint targets — all four must hit:**
1. ≥5× throughput vs naive on Zipfian, A100
2. p99 TTFT < 500ms at concurrency=32
3. Zero KV-cache leaks under 10k-request soak
4. Cancellation visible in active set within 1 scheduler tick

**Skills:**
- `engineering-skills:senior-ml-engineer` — paged attention math
- `engineering-advanced-skills:performance-profiler` — before/after paged migration
- `engineering-advanced-skills:observability-designer` — Grafana dashboard for README screenshot
- `superpowers:executing-plans` — drive M4 systematically
- `superpowers:verification-before-completion` — before claiming each of the four targets met
- `superpowers:finishing-a-development-branch` — at end

## Test strategy

| Layer | Approach | Proof |
|---|---|---|
| `block_manager`, `policy`, `sampling`, SSE formatter, queue admission | Pure TDD — fast unit tests, no torch | `pytest tests/unit/` < 2s |
| Forward-pass correctness | Integration parity — our output == HF `generate()` for fixed seeds | `test_forward_parity.py`, `test_paged_attention.py` |
| Continuous batching, cancellation | Integration tests with small model on CPU | `test_continuous_batching.py`, `test_cancellation.py` |
| KV-cache leak | Soak — 10k requests, pool returns to full free | `test_no_kv_leak.py` (marked `slow`) |
| Throughput, TTFT, ITL | Bench-only on A100 (not in CI) | `bench/results/*.json` checked in |

**Rule:** scheduler logic with fake executor + fake clock = unit test. Anything touching real model weights = integration test, gated behind `pytest -m "not slow"` for fast loops.

## Benchmark methodology

- **Baselines:** (a) naive sequential = M1 `hf_baseline`; (b) static batching (batch=8, drains fully before next admission). Three lines on every plot.
- **Models:** Qwen 2.5 0.5B (dev iteration), **SmolLM2 1.7B Instruct** (headline numbers — open + GQA + modern + fits A100 with KV pool room). Llama 3.2 1B opportunistic.
- **Hardware:** single A100 40GB on Modal/Runpod, CUDA 12.4, `attn_implementation="flash_attention_2"`, bf16.
- **Workload:** `bench/workloads/zipfian.py` — prompt lengths Zipf(α=1.2, max=512), output lengths uniform[32, 256], arrivals Poisson(λ tunable). Plus a fixed-seed ShareGPT snapshot for empirical realism.
- **Metrics:** decode tokens/sec (server-side throughput), TTFT p50/p95/p99, ITL p50/p95/p99, GPU memory util, scheduler queue depth over time.
- **Reproducibility:** `make bench` runs the entire sweep, writes JSON + PNGs to `bench/results/`, renders README plots. Seed pinned. Hardware + commit SHA stamped into output JSON.

## CV bullet ↔ milestone mapping

| Bullet | Unlocked at | Evidence |
|---|---|---|
| "Built continuous-batching scheduler with token-budget admission, achieving 5× throughput vs sequential" | M3 + M4 | `scheduler/`, `bench/results/throughput.png` |
| "Implemented paged KV-cache with block-table attention; zero leaks over 10k-request soak" | M4 | `cache/`, `test_no_kv_leak.py` |
| "Streaming SSE API with sub-tick cancellation for AI inference" | M1 baseline + M3 cancel + M4 polish | `api/sse.py`, `test_cancellation.py` |
| "Reproducible benchmark harness with Zipfian workload, Prometheus + Grafana observability" | M1 + M4 sweep | `bench/`, `grafana/dashboards/`, `make bench` |

## Risks + mitigations

1. **Llama 3.2 gating delays** → apply day 1; defaults Qwen 0.5B + SmolLM2 1.7B are both ungated. Llama is upside, not blocker.
2. **HF `transformers` API drift** → pinned `4.46.3`. Parity test in M2 catches silent semantic changes if you ever bump.
3. **A100 hardware variance across rental sessions** → 3 runs per data point, report median + IQR, stamp commit SHA + GPU model into result JSON. Don't compare across providers.
4. **KV-cache leak debugging** → every alloc/free goes through one `BlockManager` method with `metrics.kv_blocks_in_use` gauge updated in lockstep. Soak asserts pool returns to full free; if it fails, Grafana time-series shows which request type leaked.
5. **Cancellation race conditions** → single rule: scheduler is the *only* code that mutates request state. API handler enqueues a `CANCEL` intent on a per-request `asyncio.Event`; scheduler reads at top of every tick. No locks, no shared mutable state across the asyncio/torch boundary.
6. **flash-attn install hell** → pin `flash-attn==2.7.0.post2`, install with `--no-build-isolation`, lock the CUDA + torch + flash-attn triple. Eager-attention fallback always works so a flash-attn break doesn't block bench day.
7. **CPU dev too slow for continuous-batching iteration** → Qwen 0.5B in bf16 on CPU does ~5-10 tok/s, enough for 32-request integration tests in <30s. If it bites, drop to `gpt2-medium` for unit/integration only and keep the real model for bench.

## Critical files to be created

- `C:\Users\anura\Downloads\Devlopment\video-automation\continuous-batching-server\src\cbserver\scheduler\scheduler.py`
- `C:\Users\anura\Downloads\Devlopment\video-automation\continuous-batching-server\src\cbserver\cache\block_manager.py`
- `C:\Users\anura\Downloads\Devlopment\video-automation\continuous-batching-server\src\cbserver\model\executor.py`
- `C:\Users\anura\Downloads\Devlopment\video-automation\continuous-batching-server\src\cbserver\engine\engine.py`
- `C:\Users\anura\Downloads\Devlopment\video-automation\continuous-batching-server\bench\harness.py`

## Stretch goals (post-M4, only if time allows)

- INT8 weight-only quantization via `bitsandbytes` — ~2× more concurrent sequences per GPU; one config flag, one extra benchmark line.
- Speculative decoding with Qwen 0.5B as draft for SmolLM2 1.7B — research-flavored extension visible in TTFT/ITL plots.
- Prefix caching (block sharing across requests with common prompt prefix) — natural extension once refcounting works.
- Fairness algorithm comparison — swap FCFS for VTC (Virtual Token Counter) or shortest-remaining-first; plot p99 latency under heavy load.
- Multi-GPU tensor parallelism — listed in README as "deferred" only; would require sharding the KV pool across ranks.

## Verification — how to test end-to-end

After each milestone:

**M1:**
```bash
cd ~/projects/continuous-batching-server  # WSL2
make dev                                   # uvicorn server up
curl -N -X POST localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello","max_tokens":32,"stream":true}'
# Should stream tokens via SSE
make test                                  # all tests/unit/ + tests/e2e/ pass
make bench-naive CONCURRENCY=8             # produces bench/results/m1_naive.json
```

**M2:**
```bash
pytest tests/integration/test_forward_parity.py -v
# Must pass — our manual loop produces identical tokens to HF generate() at temperature=0
```

**M3:**
```bash
pytest tests/integration/test_continuous_batching.py -v
make bench-batched CONCURRENCY=32
# bench/results/m3_batched.json shows ≥3× throughput vs m1_naive
```

**M4 (final acceptance — all four must pass):**
```bash
pytest tests/integration/test_paged_attention.py -v        # paged == padded
pytest tests/integration/test_no_kv_leak.py -m slow -v     # 10k soak, no leak
pytest tests/integration/test_cancellation.py -v           # cancel in 1 tick

# Provision A100 on Runpod/Modal, then:
make bench MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct
# Inspect bench/results/sweep.json — must show:
#   throughput ratio vs naive ≥ 5.0
#   ttft_p99_ms at concurrency=32 < 500
# Inspect bench/results/throughput.png and latency.png — committed for the README
```

When all four checkpoint targets pass with evidence in `bench/results/`, the four CV bullets are honestly defensible. Update `cv.tex` with the bullets phrased verbatim from the table above (each ≤120 chars per the Jake template line-length budget) and the project is ready to ship.
