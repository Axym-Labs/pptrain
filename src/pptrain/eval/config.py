from __future__ import annotations

from typing import Any, Callable, Mapping

from pptrain.eval.harness import EvalHarness
from pptrain.eval.tasks.arc_agi2 import ARCAGI2TextTask
from pptrain.eval.tasks.bigbench import BigBenchJsonTask
from pptrain.eval.tasks.gsm8k import GSM8KTask
from pptrain.eval.tasks.humaneval import HumanEvalTask
from pptrain.eval.tasks.perplexity import PerplexityTask

TaskFactory = Callable[[Mapping[str, Any]], object]


def _build_perplexity(config: Mapping[str, Any]) -> PerplexityTask:
    return PerplexityTask(
        dataset_name=str(config["dataset_name"]),
        dataset_config_name=config.get("dataset_config_name"),
        split=str(config.get("split", "validation[:128]")),
        text_field=str(config.get("text_field", "text")),
        max_length=int(config.get("max_length", 256)),
    )


def _build_gsm8k(config: Mapping[str, Any]) -> GSM8KTask:
    fewshot_examples = [tuple(item) for item in config.get("fewshot_examples", [])]
    return GSM8KTask(
        split=str(config.get("split", "test[:32]")),
        dataset_name=str(config.get("dataset_name", "openai/gsm8k")),
        subset=str(config.get("subset", "main")),
        max_new_tokens=int(config.get("max_new_tokens", 128)),
        fewshot_examples=fewshot_examples,
    )


def _build_human_eval(config: Mapping[str, Any]) -> HumanEvalTask:
    return HumanEvalTask(
        dataset_path=str(config["dataset_path"]),
        max_examples=int(config["max_examples"]) if "max_examples" in config else None,
        max_new_tokens=int(config.get("max_new_tokens", 256)),
    )


def _build_bigbench_json(config: Mapping[str, Any]) -> BigBenchJsonTask:
    return BigBenchJsonTask(
        task_path=str(config["task_path"]),
        max_examples=int(config["max_examples"]) if "max_examples" in config else None,
        max_new_tokens=int(config.get("max_new_tokens", 32)),
    )


def _build_arc_agi2_text(config: Mapping[str, Any]) -> ARCAGI2TextTask:
    return ARCAGI2TextTask(
        data_dir=str(config["data_dir"]),
        max_tasks=int(config["max_tasks"]) if "max_tasks" in config else None,
        max_new_tokens=int(config.get("max_new_tokens", 256)),
    )


TASK_FACTORIES: dict[str, TaskFactory] = {
    "arc_agi2_text": _build_arc_agi2_text,
    "bigbench_json": _build_bigbench_json,
    "gsm8k": _build_gsm8k,
    "human_eval": _build_human_eval,
    "perplexity": _build_perplexity,
}


def build_eval_harness(config: Mapping[str, Any]) -> EvalHarness:
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("eval config must contain a non-empty 'tasks' list.")
    built_tasks = []
    for item in tasks:
        if not isinstance(item, Mapping):
            raise ValueError("Each eval task must be a mapping.")
        task_type = item.get("type")
        if not isinstance(task_type, str):
            raise ValueError("Each eval task must include a string 'type'.")
        if task_type not in TASK_FACTORIES:
            raise KeyError(f"Unsupported eval task '{task_type}'. Registered: {sorted(TASK_FACTORIES)}")
        built_tasks.append(TASK_FACTORIES[task_type](item))
    return EvalHarness(built_tasks)
