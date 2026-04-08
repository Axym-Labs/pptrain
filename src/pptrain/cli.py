from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from pptrain.core.config import RunConfig
from pptrain.core.registry import registered_mechanisms
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


def _fit_summary(trainer: PrePreTrainer, run) -> dict[str, Any]:
    return {
        "mechanism": trainer.mechanism.name,
        "run_dir": str(run.run_dir),
        "model_dir": str(run.model_dir),
        "plot_path": str(run.plot_path) if run.plot_path is not None else None,
        "metrics": run.metrics,
    }


def _print_fit_summary(summary: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"mechanism: {summary['mechanism']}")
    print(f"run_dir: {summary['run_dir']}")
    print(f"model_dir: {summary['model_dir']}")
    if summary["plot_path"] is not None:
        print(f"plot_path: {summary['plot_path']}")
    metrics = summary["metrics"]
    if metrics:
        print("metrics:")
        for name, value in sorted(metrics.items()):
            print(f"  {name}: {value}")


def _print_mechanisms(*, json_output: bool) -> None:
    mechanisms = [
        {"name": item.name, "description": item.description}
        for item in registered_mechanisms()
    ]
    if json_output:
        print(json.dumps(mechanisms, indent=2))
        return
    width = max((len(item["name"]) for item in mechanisms), default=0)
    for item in mechanisms:
        print(f"{item['name']:<{width}}  {item['description']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pptrain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Run pre-pre-training from a YAML config.")
    fit_parser.add_argument("config", type=Path)
    fit_parser.add_argument("--json", action="store_true", help="Print run summary as JSON.")

    mechanisms_parser = subparsers.add_parser(
        "mechanisms",
        help="List registered upstream mechanisms.",
    )
    mechanisms_parser.add_argument("--json", action="store_true", help="Print mechanism info as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fit":
        trainer = _build_trainer(_load_yaml(args.config))
        run = trainer.fit()
        _print_fit_summary(_fit_summary(trainer, run), json_output=args.json)
    elif args.command == "mechanisms":
        _print_mechanisms(json_output=args.json)


if __name__ == "__main__":
    main()
