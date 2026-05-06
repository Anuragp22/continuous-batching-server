import asyncio
import json
import math
import os
import platform
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import httpx

from bench.workloads.zipfian import make_prompts, zipfian_prompt_lengths


@dataclass
class RequestResult:
    prompt_length_words: int
    max_tokens: int
    ttft_ms: float
    total_ms: float
    completion_chunks: int
    success: bool
    error: str | None = None


async def _stream_one_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int,
    model: str,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    }
    request_start = time.monotonic()
    ttft_ms: float | None = None
    chunks = 0

    try:
        async with client.stream(
            "POST", f"{base_url}/v1/completions", json=payload
        ) as response:
            response.raise_for_status()
            stream_start = time.monotonic()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload_text = line[len("data: ") :]
                if payload_text == "[DONE]":
                    break
                if ttft_ms is None:
                    ttft_ms = (time.monotonic() - stream_start) * 1000.0
                chunks += 1

        total_ms = (time.monotonic() - request_start) * 1000.0
        return RequestResult(
            prompt_length_words=len(prompt.split()),
            max_tokens=max_tokens,
            ttft_ms=ttft_ms or 0.0,
            total_ms=total_ms,
            completion_chunks=max(0, chunks - 1),
            success=True,
        )
    except (httpx.HTTPError, httpx.StreamError, OSError) as exc:
        return RequestResult(
            prompt_length_words=len(prompt.split()),
            max_tokens=max_tokens,
            ttft_ms=0.0,
            total_ms=(time.monotonic() - request_start) * 1000.0,
            completion_chunks=0,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_workload(
    base_url: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
    model: str,
    timeout_s: float = 600.0,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async with httpx.AsyncClient(timeout=timeout_s) as client:

        async def _bounded(prompt: str) -> None:
            async with semaphore:
                results.append(
                    await _stream_one_request(client, base_url, prompt, max_tokens, model)
                )

        await asyncio.gather(*(_bounded(p) for p in prompts))

    return results


def percentile(values: Iterable[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation).

    Returns the value at rank ceil(p/100 * n) using 1-based indexing, then
    converted to 0-based via subtraction. So percentile([10, 20], 50) == 10
    and percentile([10, 20], 99) == 20. Avoids interpolation artifacts in
    JSON diffs across runs.
    """
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    if not 0.0 <= p <= 100.0:
        raise ValueError("percentile p must be in [0, 100]")
    rank = math.ceil(p / 100.0 * len(sorted_values))
    idx = max(0, rank - 1)
    return sorted_values[idx]


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _hardware_stamp() -> dict[str, object]:
    info: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
        "gpu": "none",
    }
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def summarize(
    results: list[RequestResult],
    wall_time_s: float,
    *,
    mode: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    seed: int,
) -> dict[str, object]:
    successful = [r for r in results if r.success]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    totals = [r.total_ms for r in successful]
    chunks_total = sum(r.completion_chunks for r in successful)

    return {
        "mode": mode,
        "model": model,
        "concurrency": concurrency,
        "max_tokens_per_request": max_tokens,
        "seed": seed,
        "n_requests": len(results),
        "n_successful": len(successful),
        "n_failed": len(results) - len(successful),
        "wall_time_s": round(wall_time_s, 3),
        "throughput_chunks_per_s": round(chunks_total / wall_time_s, 2)
        if wall_time_s > 0
        else 0.0,
        "ttft_ms_p50": round(percentile(ttfts, 50), 2),
        "ttft_ms_p95": round(percentile(ttfts, 95), 2),
        "ttft_ms_p99": round(percentile(ttfts, 99), 2),
        "total_ms_p50": round(percentile(totals, 50), 2),
        "total_ms_p95": round(percentile(totals, 95), 2),
        "total_ms_p99": round(percentile(totals, 99), 2),
        "commit_sha": _git_sha(),
        "timestamp": int(time.time()),
        "hardware": _hardware_stamp(),
    }


@click.command()
@click.option("--mode", required=True, type=click.Choice(["naive", "batched", "sweep"]))
@click.option("--model", required=True, help="Model identifier (echoed into result JSON).")
@click.option("--base-url", default="http://localhost:8000", show_default=True)
@click.option("--concurrency", default=8, type=int, show_default=True)
@click.option("--n-requests", default=64, type=int, show_default=True)
@click.option("--max-tokens", default=128, type=int, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--alpha", default=1.2, type=float, show_default=True)
@click.option("--max-prompt-len", default=512, type=int, show_default=True)
@click.option("--output", required=True, type=click.Path(dir_okay=False))
def main(
    mode: str,
    model: str,
    base_url: str,
    concurrency: int,
    n_requests: int,
    max_tokens: int,
    seed: int,
    alpha: float,
    max_prompt_len: int,
    output: str,
) -> None:
    lengths = zipfian_prompt_lengths(
        n_requests, alpha=alpha, max_len=max_prompt_len, seed=seed
    )
    prompts = make_prompts(lengths, seed=seed)

    click.echo(
        f"Running {mode} workload: {n_requests} requests, "
        f"concurrency={concurrency}, max_tokens={max_tokens}"
    )
    start = time.monotonic()
    results = asyncio.run(
        run_workload(base_url, prompts, max_tokens, concurrency, model)
    )
    wall_time = time.monotonic() - start

    summary = summarize(
        results,
        wall_time,
        mode=mode,
        model=model,
        concurrency=concurrency,
        max_tokens=max_tokens,
        seed=seed,
    )
    summary["per_request"] = [asdict(r) for r in results]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    click.echo(
        f"Throughput: {summary['throughput_chunks_per_s']:.1f} chunks/s | "
        f"TTFT p50/p95/p99: "
        f"{summary['ttft_ms_p50']:.1f} / "
        f"{summary['ttft_ms_p95']:.1f} / "
        f"{summary['ttft_ms_p99']:.1f} ms | "
        f"failed: {summary['n_failed']}/{summary['n_requests']}"
    )
    click.echo(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
