from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pptrain.eval.base import EvalResult, EvalTask
from pptrain.eval.generation import generate_text
from pptrain.eval.tasks.perplexity import _require_datasets

ANSWER_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")


def extract_final_number(text: str) -> str | None:
    matches = ANSWER_RE.findall(text.replace("$", ""))
    if not matches:
        return None
    return matches[-1].replace(",", "")


@dataclass(slots=True)
class GSM8KTask(EvalTask):
    split: str = "test[:32]"
    dataset_name: str = "openai/gsm8k"
    subset: str = "main"
    max_new_tokens: int = 128
    fewshot_examples: list[tuple[str, str]] = field(default_factory=list)
    name: str = "gsm8k"

    def run(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **_: Any,
    ) -> EvalResult:
        load_dataset = _require_datasets()
        dataset = load_dataset(self.dataset_name, self.subset, split=self.split)
        correct = 0
        total = 0
        for record in dataset:
            prompt = self._build_prompt(record["question"])
            prediction = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
            )
            predicted = extract_final_number(prediction)
            gold = extract_final_number(record["answer"])
            correct += int(predicted is not None and gold is not None and predicted == gold)
            total += 1
        accuracy = correct / max(total, 1)
        return EvalResult(
            name=self.name,
            metrics={"accuracy": accuracy},
            artifacts={"num_examples": total},
        )

    def _build_prompt(self, question: str) -> str:
        lines = ["Solve the math word problem and end with 'Answer: <number>'."]
        for example_question, example_answer in self.fewshot_examples:
            lines.append(f"Question: {example_question}\nAnswer: {example_answer}")
        lines.append(f"Question: {question}\nAnswer:")
        return "\n\n".join(lines)

