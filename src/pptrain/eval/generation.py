from __future__ import annotations

from typing import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def resolve_device(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    device = resolve_device(model)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id or model.config.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = max(temperature, 1e-5)
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    generated = outputs[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def score_multiple_choice(prediction: str, choices: Sequence[str], target_scores: dict[str, float]) -> float:
    normalized = prediction.strip().lower()
    for choice in choices:
        normalized_choice = choice.strip().lower()
        if normalized == normalized_choice or normalized.startswith(normalized_choice):
            return float(target_scores.get(choice, 0.0))
    return 0.0
