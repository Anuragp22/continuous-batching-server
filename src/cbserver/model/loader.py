from dataclasses import dataclass
from typing import Any


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str
    dtype: str


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    return "bfloat16" if device == "cuda" else "float32"


def load_model_and_tokenizer(
    model_id: str,
    device: str = "auto",
    dtype: str = "auto",
    attn_implementation: str = "eager",
) -> LoadedModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    torch_dtype = getattr(torch, resolved_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    ).to(resolved_device)
    model.train(False)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        dtype=resolved_dtype,
    )
