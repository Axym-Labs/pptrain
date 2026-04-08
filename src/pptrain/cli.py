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
from pptrain.eval.runner import run_transfer_evaluation
from pptrain.integrations import HFCausalLMAdapter, HFModelConfig
from pptrain.replication import run_replication_campaign


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


def _fit_summary(
    trainer: PrePreTrainer,
    run,
    *,
    eval_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "mechanism": trainer.mechanism.name,
        "run_dir": str(run.run_dir),
        "model_dir": str(run.model_dir),
        "plot_path": str(run.plot_path) if run.plot_path is not None else None,
        "eval_path": str(eval_path) if eval_path is not None else None,
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
    if summary["eval_path"] is not None:
        print(f"eval_path: {summary['eval_path']}")
    metrics = summary["metrics"]
    if metrics:
        print("metrics:")
        for name, value in sorted(metrics.items()):
            print(f"  {name}: {value}")


def _print_mechanisms(*, json_output: bool, mechanism_name: str | None = None) -> None:
    mechanisms = [
        {
            "name": item.name,
            "description": item.description,
            "presets": [
                {
                    "name": preset.name,
                    "description": preset.description,
                    "reference": preset.reference,
                }
                for preset in item.presets
            ],
        }
        for item in registered_mechanisms()
    ]
    if mechanism_name is not None:
        mechanisms = [item for item in mechanisms if item["name"] == mechanism_name]
        if not mechanisms:
            raise KeyError(f"Unknown mechanism '{mechanism_name}'.")
    if json_output:
        print(json.dumps(mechanisms, indent=2))
        return
    width = max((len(item["name"]) for item in mechanisms), default=0)
    for item in mechanisms:
        print(f"{item['name']:<{width}}  {item['description']}")
        presets = item["presets"]
        if presets:
            preset_names = ", ".join(preset["name"] for preset in presets)
            print(f"{'':<{width}}  presets: {preset_names}")


def _maybe_run_eval(args: argparse.Namespace, trainer: PrePreTrainer, run) -> Path | None:
    if args.eval_config is None:
        return None
    eval_config = _load_yaml(args.eval_config)
    return run_transfer_evaluation(
        bundle=run.load_transfer_bundle(),
        model_adapter=trainer.model_adapter,
        eval_config=eval_config,
        output_dir=run.run_dir / "eval",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pptrain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Run pre-pre-training from a YAML config.")
    fit_parser.add_argument("config", type=Path)
    fit_parser.add_argument(
        "--eval-config",
        type=Path,
        help="Optional YAML config for a lightweight post-transfer evaluation pass.",
    )
    fit_parser.add_argument("--json", action="store_true", help="Print run summary as JSON.")

    mechanisms_parser = subparsers.add_parser(
        "mechanisms",
        help="List registered upstream mechanisms.",
    )
    mechanisms_parser.add_argument("name", nargs="?", help="Optional mechanism name filter.")
    mechanisms_parser.add_argument("--json", action="store_true", help="Print mechanism info as JSON.")

    replicate_parser = subparsers.add_parser(
        "replicate",
        help="Run the built-in replication or paper-proxy campaign.",
    )
    replicate_parser.add_argument(
        "--profile",
        default="paper_proxy_2048",
        help="Replication profile name. Use 'smoke' or 'paper_proxy_2048'.",
    )
    replicate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/replication"),
        help="Directory to store replication artifacts.",
    )
    replicate_parser.add_argument(
        "--test",
        action="store_true",
        help="Run the very low-compute smoke validation profile.",
    )
    replicate_parser.add_argument(
        "--mechanism",
        action="append",
        dest="mechanisms",
        help="Optional mechanism name filter. Can be passed multiple times.",
    )
    replicate_parser.add_argument(
        "--model-name-or-path",
        help="Optional model override for the replication campaign.",
    )
    replicate_parser.add_argument(
        "--context-length",
        type=int,
        help="Optional context-length override for the replication campaign.",
    )
    replicate_parser.add_argument(
        "--seeds",
        help="Optional comma-separated seed list for the replication campaign.",
    )
    replicate_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially completed replication campaign from replication_results.json.",
    )
    replicate_parser.add_argument(
        "--keep-checkpoints",
        action="store_false",
        dest="remove_checkpoints",
        help="Keep intermediate Trainer checkpoint-* directories instead of pruning them after final model save.",
    )
    replicate_parser.set_defaults(remove_checkpoints=True)
    replicate_parser.add_argument("--json", action="store_true", help="Print the top-level campaign payload as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fit":
        trainer = _build_trainer(_load_yaml(args.config))
        run = trainer.fit()
        eval_path = _maybe_run_eval(args, trainer, run)
        _print_fit_summary(_fit_summary(trainer, run, eval_path=eval_path), json_output=args.json)
    elif args.command == "mechanisms":
        _print_mechanisms(json_output=args.json, mechanism_name=args.name)
    elif args.command == "replicate":
        seed_values = None
        if args.seeds:
            seed_values = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
        payload = run_replication_campaign(
            profile_name=args.profile,
            output_dir=str(args.output_dir),
            test_mode=bool(args.test),
            mechanisms=args.mechanisms,
            model_name_or_path=args.model_name_or_path,
            context_length=args.context_length,
            seeds=seed_values,
            remove_checkpoints=bool(args.remove_checkpoints),
            resume=bool(args.resume),
        )
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"profile: {payload['profile']['name']}")
            print(f"output_dir: {args.output_dir}")
            print("artifacts:")
            for name, path in sorted(payload.get("artifacts", {}).items()):
                print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
