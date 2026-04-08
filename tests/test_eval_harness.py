from pathlib import Path

from pptrain.eval import EvalHarness, EvalResult, EvalTask


class DummyTask(EvalTask):
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.value = value

    def run(self, **kwargs):
        return EvalResult(name=self.name, metrics={"score": self.value})


def test_eval_harness_run_and_save(tmp_path: Path) -> None:
    harness = EvalHarness([DummyTask("alpha", 0.5), DummyTask("beta", 0.75)])
    results = harness.run_and_save(str(tmp_path))
    assert set(results) == {"alpha", "beta"}
    assert (tmp_path / "eval_results.json").exists()
    assert (tmp_path / "eval_summary.png").exists()

