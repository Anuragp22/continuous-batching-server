import pytest

from bench.workloads.zipfian import make_prompts, zipfian_prompt_lengths


def test_lengths_within_bounds() -> None:
    lengths = zipfian_prompt_lengths(200, alpha=1.2, max_len=512, seed=42)
    assert len(lengths) == 200
    assert all(1 <= length <= 512 for length in lengths)


def test_lengths_are_seed_deterministic() -> None:
    a = zipfian_prompt_lengths(100, alpha=1.2, max_len=512, seed=7)
    b = zipfian_prompt_lengths(100, alpha=1.2, max_len=512, seed=7)
    c = zipfian_prompt_lengths(100, alpha=1.2, max_len=512, seed=8)
    assert a == b
    assert a != c


def test_zero_count_returns_empty() -> None:
    assert zipfian_prompt_lengths(0, seed=0) == []


def test_invalid_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        zipfian_prompt_lengths(10, alpha=1.0)


def test_invalid_max_len_rejected() -> None:
    with pytest.raises(ValueError):
        zipfian_prompt_lengths(10, max_len=0)


def test_distribution_is_zipf_shaped() -> None:
    lengths = zipfian_prompt_lengths(2000, alpha=1.2, max_len=512, seed=42)
    short_count = sum(1 for length in lengths if length <= 4)
    long_count = sum(1 for length in lengths if length > 100)
    assert short_count > long_count


def test_make_prompts_word_counts_match_lengths() -> None:
    lengths = [3, 7, 12]
    prompts = make_prompts(lengths, seed=1)
    assert [len(p.split()) for p in prompts] == lengths


def test_make_prompts_seed_determinism() -> None:
    a = make_prompts([5, 5], seed=42)
    b = make_prompts([5, 5], seed=42)
    c = make_prompts([5, 5], seed=43)
    assert a == b
    assert a != c
