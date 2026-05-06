import pytest

from bench.harness import RequestResult, percentile, summarize


def test_percentile_picks_floor_index_for_small_samples() -> None:
    assert percentile([10.0, 20.0], 50) == 10.0
    assert percentile([10.0, 20.0], 99) == 20.0


def test_percentile_empty_returns_zero() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([], 99) == 0.0


def test_percentile_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], -1.0)
    with pytest.raises(ValueError):
        percentile([1.0], 101.0)


def test_percentile_monotonic_in_p() -> None:
    values = [1.0, 5.0, 10.0, 50.0, 100.0]
    assert percentile(values, 50) <= percentile(values, 95)
    assert percentile(values, 95) <= percentile(values, 99)


def _result(
    *, success: bool = True, ttft_ms: float = 50.0, total_ms: float = 200.0, chunks: int = 32
) -> RequestResult:
    return RequestResult(
        prompt_length_words=10,
        max_tokens=128,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        completion_chunks=chunks,
        success=success,
        error=None if success else "boom",
    )


def test_summary_counts_success_and_failure() -> None:
    results = [_result(), _result(), _result(success=False, ttft_ms=0.0, chunks=0)]
    summary = summarize(
        results,
        wall_time_s=10.0,
        mode="naive",
        model="qwen",
        concurrency=4,
        max_tokens=128,
        seed=42,
    )
    assert summary["n_requests"] == 3
    assert summary["n_successful"] == 2
    assert summary["n_failed"] == 1


def test_summary_throughput_excludes_failed_requests() -> None:
    results = [
        _result(chunks=64),
        _result(chunks=64),
        _result(success=False, chunks=0),
    ]
    summary = summarize(
        results,
        wall_time_s=2.0,
        mode="naive",
        model="qwen",
        concurrency=2,
        max_tokens=128,
        seed=0,
    )
    assert summary["throughput_chunks_per_s"] == pytest.approx(64.0)


def test_summary_records_metadata() -> None:
    summary = summarize(
        [_result()],
        wall_time_s=1.0,
        mode="batched",
        model="smollm2",
        concurrency=8,
        max_tokens=256,
        seed=99,
    )
    assert summary["mode"] == "batched"
    assert summary["model"] == "smollm2"
    assert summary["concurrency"] == 8
    assert summary["max_tokens_per_request"] == 256
    assert summary["seed"] == 99
    assert "commit_sha" in summary
    assert "hardware" in summary
    assert "timestamp" in summary


def test_summary_handles_zero_wall_time() -> None:
    summary = summarize(
        [_result()],
        wall_time_s=0.0,
        mode="naive",
        model="qwen",
        concurrency=1,
        max_tokens=128,
        seed=0,
    )
    assert summary["throughput_chunks_per_s"] == 0.0


def test_summary_with_all_failed_requests_reports_zeros() -> None:
    results = [_result(success=False, ttft_ms=0.0, chunks=0) for _ in range(5)]
    summary = summarize(
        results,
        wall_time_s=2.5,
        mode="naive",
        model="qwen",
        concurrency=4,
        max_tokens=128,
        seed=0,
    )
    assert summary["n_successful"] == 0
    assert summary["n_failed"] == 5
    assert summary["throughput_chunks_per_s"] == 0.0
    assert summary["ttft_ms_p99"] == 0.0
    assert summary["total_ms_p99"] == 0.0


def test_percentile_p0_and_p100_are_min_and_max() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 40.0


def test_percentile_single_element_returns_that_element() -> None:
    assert percentile([42.0], 0) == 42.0
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 100) == 42.0
