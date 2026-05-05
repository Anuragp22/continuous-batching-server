from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

requests_total = Counter(
    "cbserver_requests_total",
    "Total completion requests received, labeled by terminal status.",
    ["status"],
    registry=REGISTRY,
)

active_requests = Gauge(
    "cbserver_active_requests",
    "Requests currently in WAITING or RUNNING state.",
    registry=REGISTRY,
)

ttft_seconds = Histogram(
    "cbserver_ttft_seconds",
    "Time-to-first-token, server-side.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=REGISTRY,
)

itl_seconds = Histogram(
    "cbserver_itl_seconds",
    "Inter-token latency, server-side.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

decode_tokens_total = Counter(
    "cbserver_decode_tokens_total",
    "Total decode tokens emitted across all requests.",
    registry=REGISTRY,
)

scheduler_queue_depth = Gauge(
    "cbserver_scheduler_queue_depth",
    "Sequences currently waiting for admission.",
    registry=REGISTRY,
)

kv_blocks_in_use = Gauge(
    "cbserver_kv_blocks_in_use",
    "Physical KV blocks currently allocated.",
    registry=REGISTRY,
)
