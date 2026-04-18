from __future__ import annotations

import sys
from importlib import import_module

from pptrain.mechanisms.dyck import DyckConfig, DyckMechanism, DyckTaskFamily
from pptrain.mechanisms.lime import LIMEConfig, LIMEMechanism, LIMETaskFamily
from pptrain.mechanisms.nca import NCAConfig, NCAMechanism, NCATask
from pptrain.mechanisms.procedural import ProceduralConfig, ProceduralMechanism, ProceduralTaskFamily
from pptrain.mechanisms.simpler_tasks import (
    SimplerTasksConfig,
    SimplerTasksMechanism,
    SimplerTasksTaskFamily,
)
from pptrain.mechanisms.summarization import (
    SummarizationConfig,
    SummarizationMechanism,
    SummarizationTaskFamily,
)

_SUBMODULE_ALIASES = {
    "dyck": "pptrain.mechanisms.dyck",
    "lime": "pptrain.mechanisms.lime",
    "nca": "pptrain.mechanisms.nca",
    "procedural": "pptrain.mechanisms.procedural",
    "simpler_tasks": "pptrain.mechanisms.simpler_tasks",
    "summarization": "pptrain.mechanisms.summarization",
}

for alias, target in _SUBMODULE_ALIASES.items():
    sys.modules[f"{__name__}.{alias}"] = import_module(target)

__all__ = [
    "DyckConfig",
    "DyckTaskFamily",
    "DyckMechanism",
    "LIMEConfig",
    "LIMETaskFamily",
    "LIMEMechanism",
    "NCAConfig",
    "NCATask",
    "NCAMechanism",
    "ProceduralConfig",
    "ProceduralTaskFamily",
    "ProceduralMechanism",
    "SimplerTasksConfig",
    "SimplerTasksTaskFamily",
    "SimplerTasksMechanism",
    "SummarizationConfig",
    "SummarizationTaskFamily",
    "SummarizationMechanism",
]
