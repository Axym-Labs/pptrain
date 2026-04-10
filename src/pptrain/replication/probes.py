from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pptrain.eval.base import EvalResult
from pptrain.eval.generation import generate_text
from pptrain.eval.tasks.gsm8k import GSM8KTask, extract_final_number
from pptrain.replication.specs import ArithmeticProbeConfig, GSM8KEvalConfig, NeedleProbeConfig


@dataclass(slots=True)
class ProbeBundle:
    reasoning: EvalResult | None = None
    algorithmic: EvalResult | None = None


def _prepare_model_for_generation(model: PreTrainedModel) -> PreTrainedModel:
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        if device.type != "cuda":
            model = model.to("cuda")
    model.eval()
    return model


def _score_candidates(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    candidates: list[str],
) -> dict[str, float]:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    scores: dict[str, float] = {}
    with torch.no_grad():
        for candidate in candidates:
            continuation = tokenizer(f" {candidate}", add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            full_ids = torch.cat([prompt_ids, continuation], dim=1)
            logits = model(full_ids).logits[:, :-1, :]
            target_ids = full_ids[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            continuation_log_probs = gathered[:, prompt_ids.shape[1] - 1 :]
            scores[candidate] = float(continuation_log_probs.sum().item())
    return scores


def _select_best_candidate(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    candidates: list[str],
) -> str:
    scores = _score_candidates(model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates)
    return max(candidates, key=lambda candidate: scores[candidate])


def run_arithmetic_probe(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: ArithmeticProbeConfig,
) -> EvalResult:
    model = _prepare_model_for_generation(model)
    rng = np.random.default_rng(42)
    correct = 0
    for _ in range(config.num_examples):
        left = int(rng.integers(1, config.max_addend + 1))
        right = int(rng.integers(1, config.max_addend + 1))
        prompt = (
            "Solve the arithmetic problem and answer with the number only.\n"
            f"Question: What is {left} + {right}?\nAnswer:"
        )
        prediction = _select_best_candidate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidates=[str(value) for value in range(2, config.max_addend * 2 + 1)],
        )
        correct += int(prediction == str(left + right))
    accuracy = correct / max(config.num_examples, 1)
    return EvalResult(
        name="arithmetic_probe",
        metrics={"accuracy": accuracy},
        artifacts={"num_examples": config.num_examples, "mode": "candidate_logprob"},
    )


def run_needle_probe(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: NeedleProbeConfig,
) -> EvalResult:
    model = _prepare_model_for_generation(model)
    rng = np.random.default_rng(13)
    values = list("123456789")
    correct = 0
    for example_index in range(config.num_examples):
        target_slot = int(rng.integers(0, config.haystack_size))
        lines = ["Remember the key value map and answer with the value only."]
        target_answer = ""
        for slot in range(config.haystack_size):
            key = f"key_{example_index}_{slot}"
            value = values[(example_index + slot) % len(values)]
            if slot == target_slot:
                target_answer = value
                target_key = key
            lines.append(f"{key}: {value}")
        lines.append(f"Question: What is the value for {target_key}?")
        lines.append("Answer:")
        prediction = _select_best_candidate(
            model=model,
            tokenizer=tokenizer,
            prompt="\n".join(lines),
            candidates=values,
        )
        correct += int(prediction == target_answer)
    accuracy = correct / max(config.num_examples, 1)
    return EvalResult(
        name="needle_probe",
        metrics={"accuracy": accuracy},
        artifacts={"num_examples": config.num_examples, "mode": "candidate_logprob"},
    )


def run_gsm8k_probe(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: GSM8KEvalConfig,
) -> EvalResult:
    model = _prepare_model_for_generation(model)
    task = GSM8KTask(
        split=config.split,
        max_new_tokens=config.max_new_tokens,
        fewshot_examples=list(config.fewshot_examples),
    )
    return task.run(model=model, tokenizer=tokenizer)
