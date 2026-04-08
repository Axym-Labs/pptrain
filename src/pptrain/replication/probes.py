from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pptrain.eval.base import EvalResult
from pptrain.eval.generation import generate_text
from pptrain.eval.tasks.gsm8k import GSM8KTask, extract_final_number
from pptrain.replication.specs import ArithmeticProbeConfig, GSM8KEvalConfig, NeedleProbeConfig


@dataclass(slots=True)
class ProbeBundle:
    reasoning: EvalResult | None = None
    algorithmic: EvalResult | None = None


def run_arithmetic_probe(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: ArithmeticProbeConfig,
) -> EvalResult:
    rng = np.random.default_rng(42)
    correct = 0
    for _ in range(config.num_examples):
        left = int(rng.integers(1, config.max_addend + 1))
        right = int(rng.integers(1, config.max_addend + 1))
        prompt = (
            "Solve the arithmetic problem and answer with the number only.\n"
            f"Question: What is {left} + {right}?\nAnswer:"
        )
        prediction = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=config.max_new_tokens,
        )
        predicted = extract_final_number(prediction)
        correct += int(predicted == str(left + right))
    accuracy = correct / max(config.num_examples, 1)
    return EvalResult(name="arithmetic_probe", metrics={"accuracy": accuracy}, artifacts={"num_examples": config.num_examples})


def run_needle_probe(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: NeedleProbeConfig,
) -> EvalResult:
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
        prediction = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="\n".join(lines),
            max_new_tokens=config.max_new_tokens,
        )
        predicted = extract_final_number(prediction)
        correct += int(predicted == target_answer)
    accuracy = correct / max(config.num_examples, 1)
    return EvalResult(name="needle_probe", metrics={"accuracy": accuracy}, artifacts={"num_examples": config.num_examples})


def run_gsm8k_probe(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: GSM8KEvalConfig,
) -> EvalResult:
    task = GSM8KTask(
        split=config.split,
        max_new_tokens=config.max_new_tokens,
        fewshot_examples=list(config.fewshot_examples),
    )
    return task.run(model=model, tokenizer=tokenizer)
