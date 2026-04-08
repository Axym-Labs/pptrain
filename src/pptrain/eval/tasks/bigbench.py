from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pptrain.eval.base import EvalResult, EvalTask
from pptrain.eval.generation import generate_text, score_multiple_choice


@dataclass(slots=True)
class BigBenchJsonTask(EvalTask):
    task_path: str
    max_examples: int | None = None
    max_new_tokens: int = 32
    name: str = "bigbench_json"

    def run(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **_: Any,
    ) -> EvalResult:
        task = json.loads(Path(self.task_path).read_text(encoding="utf-8"))
        examples = task["examples"]
        if self.max_examples is not None:
            examples = examples[: self.max_examples]
        total_score = 0.0
        for example in examples:
            prompt = self._format_example(task, example)
            prediction = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
            )
            choices = list(example["target_scores"].keys())
            total_score += score_multiple_choice(prediction, choices, example["target_scores"])
        score = total_score / max(len(examples), 1)
        return EvalResult(
            name=self.name,
            metrics={"multiple_choice_grade": score},
            artifacts={"task_name": task.get("name"), "num_examples": len(examples)},
        )

    @staticmethod
    def _format_example(task: dict[str, Any], example: dict[str, Any]) -> str:
        choice_prefix = task.get("choice_prefix", "\nchoice: ")
        prompt = example["input"]
        prompt += choice_prefix + choice_prefix.join(example["target_scores"].keys())
        return f"{prompt}\nAnswer:"

