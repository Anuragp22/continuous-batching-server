import random

import numpy as np


_FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "and a stitch in time saves nine while early bird gets the worm "
    "but actions speak louder than words and beauty is in the eye of the beholder"
).split()


def zipfian_prompt_lengths(
    n: int, alpha: float = 1.2, max_len: int = 512, seed: int | None = None
) -> list[int]:
    """Sample n prompt lengths from a truncated Zipf(alpha) distribution.

    Defaults match plan.md (Zipf alpha=1.2, max=512). Truncation is rejection
    sampling against [1, max_len]; this preserves the shape of the head where
    almost all of the probability mass lives.
    """
    if n <= 0:
        return []
    if alpha <= 1.0:
        raise ValueError("Zipf parameter alpha must be > 1.0")
    if max_len < 1:
        raise ValueError("max_len must be >= 1")

    rng = np.random.default_rng(seed)
    lengths: list[int] = []
    while len(lengths) < n:
        batch = rng.zipf(alpha, size=max(n * 2, 32))
        for value in batch:
            if 1 <= value <= max_len:
                lengths.append(int(value))
                if len(lengths) >= n:
                    break
    return lengths


def make_prompts(lengths: list[int], seed: int | None = None) -> list[str]:
    """Generate dummy whitespace-separated prompts of approximately the given
    word counts. The mock generator and HF baseline both tokenize roughly per
    word for short inputs, so word count is a workable proxy for token count
    at M1's level of fidelity.
    """
    rng = random.Random(seed)
    prompts: list[str] = []
    for length in lengths:
        prompts.append(" ".join(rng.choices(_FILLER_WORDS, k=max(1, length))))
    return prompts
