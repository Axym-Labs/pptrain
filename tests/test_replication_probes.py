from __future__ import annotations

import torch

from pptrain.replication import probes
from pptrain.replication.specs import ArithmeticProbeConfig, NeedleProbeConfig


class _DummyParameter:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)


class _DummyModel:
    def __init__(self) -> None:
        self._parameter = _DummyParameter()
        self.moved_to: str | None = None
        self.eval_called = False

    def parameters(self):
        yield self._parameter

    def to(self, device: str):
        self.moved_to = device
        self._parameter = _DummyParameter(device)
        return self

    def eval(self):
        self.eval_called = True
        return self


def test_prepare_model_for_generation_moves_cpu_model_to_cuda(monkeypatch) -> None:
    model = _DummyModel()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    prepared = probes._prepare_model_for_generation(model)  # type: ignore[attr-defined]

    assert prepared is model
    assert model.moved_to == "cuda"
    assert model.eval_called is True


def test_prepare_model_for_generation_keeps_cpu_when_cuda_unavailable(monkeypatch) -> None:
    model = _DummyModel()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    prepared = probes._prepare_model_for_generation(model)  # type: ignore[attr-defined]

    assert prepared is model
    assert model.moved_to is None
    assert model.eval_called is True


def test_run_arithmetic_probe_uses_candidate_scoring(monkeypatch) -> None:
    model = _DummyModel()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _fake_select_best_candidate(**kwargs):
        prompt = kwargs["prompt"]
        numbers = [int(part) for part in prompt.replace("?", "").split() if part.isdigit()]
        return str(numbers[0] + numbers[1])

    monkeypatch.setattr(probes, "_select_best_candidate", _fake_select_best_candidate)

    result = probes.run_arithmetic_probe(model=model, tokenizer=object(), config=ArithmeticProbeConfig(num_examples=3, max_addend=9))

    assert result.metrics["accuracy"] == 1.0
    assert result.artifacts["mode"] == "candidate_logprob"


def test_run_needle_probe_uses_candidate_scoring(monkeypatch) -> None:
    model = _DummyModel()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _fake_select_best_candidate(**kwargs):
        prompt = kwargs["prompt"]
        question_line = [line for line in prompt.splitlines() if line.startswith("Question:")][0]
        key = question_line.split()[-1].rstrip("?")
        for line in prompt.splitlines():
            if line.startswith(f"{key}: "):
                return line.split(": ", 1)[1]
        raise AssertionError("target key not found")

    monkeypatch.setattr(probes, "_select_best_candidate", _fake_select_best_candidate)

    result = probes.run_needle_probe(model=model, tokenizer=object(), config=NeedleProbeConfig(num_examples=3, haystack_size=8))

    assert result.metrics["accuracy"] == 1.0
    assert result.artifacts["mode"] == "candidate_logprob"
