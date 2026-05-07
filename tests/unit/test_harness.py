import asyncio

import pytest
from click.testing import CliRunner

import bench.harness as harness_module
from bench.harness import RequestResult, main, percentile, run_workload, summarize


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
    *,
    success: bool = True,
    ttft_ms: float = 50.0,
    total_ms: float = 200.0,
    chunks: int = 32,
    chars: int = 128,
) -> RequestResult:
    return RequestResult(
        prompt_length_words=10,
        max_tokens=128,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        completion_chunks=chunks,
        completion_chars=chars,
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
        _result(chunks=64, chars=256),
        _result(chunks=64, chars=256),
        _result(success=False, chunks=0, chars=0),
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
    assert summary["throughput_chars_per_s"] == pytest.approx(256.0)


def test_summary_reports_chars_throughput_alongside_chunks() -> None:
    """chars/s is the cross-backend comparison metric since SSE chunk count
    varies with TextIteratorStreamer's batching while characters do not."""
    results = [
        _result(chunks=10, chars=50),
        _result(chunks=10, chars=50),
    ]
    summary = summarize(
        results,
        wall_time_s=1.0,
        mode="naive",
        model="qwen",
        concurrency=2,
        max_tokens=128,
        seed=0,
    )
    assert summary["throughput_chunks_per_s"] == pytest.approx(20.0)
    assert summary["throughput_chars_per_s"] == pytest.approx(100.0)
    assert "metric_notes" in summary


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


def test_run_workload_preserves_input_order_under_jitter(monkeypatch) -> None:
    """Result list is indexed by input prompt position even when individual
    requests complete out-of-order (Zipfian workloads always do this).
    The sleep duration is encoded in the prompt so the fake completes in
    a non-trivially different order than input ordering."""

    async def _fake_stream(_client, _base_url, prompt, _max_tokens, _model):
        await asyncio.sleep(int(prompt.split()[0]) / 1000.0)
        return RequestResult(
            prompt_length_words=len(prompt.split()),
            max_tokens=128,
            ttft_ms=1.0,
            total_ms=1.0,
            completion_chunks=1,
            completion_chars=len(prompt),
            success=True,
            error=prompt,
        )

    monkeypatch.setattr(harness_module, "_stream_one_request", _fake_stream)

    prompts = [f"{ms} marker" for ms in (50, 5, 30, 1, 20, 10)]
    results = asyncio.run(
        run_workload("http://x", prompts, max_tokens=128, concurrency=6, model="m")
    )
    assert [r.error for r in results] == prompts


def test_run_workload_rejects_zero_concurrency() -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            run_workload("http://x", ["p"], max_tokens=8, concurrency=0, model="m")
        )


def test_run_workload_handles_empty_prompts() -> None:
    results = asyncio.run(
        run_workload("http://x", [], max_tokens=8, concurrency=4, model="m")
    )
    assert results == []


def test_cli_rejects_zero_concurrency(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--mode", "naive",
            "--model", "qwen",
            "--concurrency", "0",
            "--n-requests", "1",
            "--output", str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code != 0


def test_cli_rejects_zero_n_requests(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--mode", "naive",
            "--model", "qwen",
            "--n-requests", "0",
            "--output", str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code != 0


def test_cli_rejects_zero_max_tokens(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--mode", "naive",
            "--model", "qwen",
            "--max-tokens", "0",
            "--n-requests", "1",
            "--output", str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code != 0
