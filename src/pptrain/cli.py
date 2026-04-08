from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from pptrain.core.config import RunConfig
from pptrain.core.runner import PrePreTrainer
from pptrain.core.registry import create_mechanism
from pptrain.integrations import HFCausalLMAdapter, HFModelConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_trainer(config: dict[str, Any]) -> PrePreTrainer:
    mechanism_block = config["mechanism"]
    mechanism = create_mechanism(
        mechanism_block["name"],
        mechanism_block.get("config", {}),
    )
    model = HFCausalLMAdapter(HFModelConfig(**config["model"]))
    run_config = RunConfig(**config["run"])
    return PrePreTrainer(
        mechanism=mechanism,
        model_adapter=model,
        run_config=run_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="pptrain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Run pre-pre-training from a YAML config.")
    fit_parser.add_argument("config", type=Path)

    args = parser.parse_args()
    if args.command == "fit":
        trainer = _build_trainer(_load_yaml(args.config))
        trainer.fit()

