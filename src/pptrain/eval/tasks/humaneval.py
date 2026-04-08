from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pptrain.eval.base import EvalResult, EvalTask
from pptrain.eval.generation import generate_text


@dataclass(slots=True)
class HumanEvalTask(EvalTask):
    dataset_path: str
    max_examples: int | None = None
    max_new_tokens: int = 256
    name: str = "human_eval"

    def run(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **_: Any,
    ) -> EvalResult:
        problems = self._load_problems(Path(self.dataset_path))
        if self.max_examples is not None:
            problems = problems[: self.max_examples]
        completions = []
        for problem in problems:
            completion = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=problem["prompt"],
                max_new_tokens=self.max_new_tokens,
            )
            completions.append(
                {
                    "task_id": problem["task_id"],
                    "completion": completion,
                }
            )
        pass_at_1 = self._maybe_score(completions, Path(self.dataset_path))
        return EvalResult(
            name=self.name,
            metrics={"pass@1": pass_at_1},
            artifacts={"num_examples": len(completions)},
        )

    @staticmethod
    def _load_problems(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _maybe_score(completions: list[dict[str, Any]], dataset_path: Path) -> float:
        try:
            from human_eval.evaluation import evaluate_functional_correctness
        except ImportError:
            return 0.0

        with TemporaryDirectory() as temp_dir:
            sample_file = Path(temp_dir) / "samples.jsonl"
            sample_file.write_text(
                "\n".join(json.dumps(item) for item in completions),
                encoding="utf-8",
            )
            results = evaluate_functional_correctness(
                sample_file=str(sample_file),
                problem_file=str(dataset_path),
                k=[1],
                n_workers=1,
                timeout=3.0,
            )
        return float(results.get("pass@1", 0.0))

