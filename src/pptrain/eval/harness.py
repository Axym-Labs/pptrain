from __future__ import annotations

from typing import Any, Iterable

from pptrain.eval.base import EvalResult, EvalTask
from pptrain.eval.plotting import save_eval_summary


class EvalHarness:
    def __init__(self, tasks: Iterable[EvalTask]) -> None:
        self.tasks = list(tasks)

    def run(self, **kwargs: Any) -> dict[str, EvalResult]:
        results: dict[str, EvalResult] = {}
        for task in self.tasks:
            results[task.name] = task.run(**kwargs)
        return results

    def run_and_save(self, output_dir: str, **kwargs: Any) -> dict[str, EvalResult]:
        results = self.run(**kwargs)
        save_eval_summary(results, output_dir)
        return results
