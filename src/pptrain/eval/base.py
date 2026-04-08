from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalResult:
    name: str
    metrics: dict[str, float]
    artifacts: dict[str, Any] = field(default_factory=dict)


class EvalTask(ABC):
    name: str

    @abstractmethod
    def run(self, **kwargs: Any) -> EvalResult:
        raise NotImplementedError

