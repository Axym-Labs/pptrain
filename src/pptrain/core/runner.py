from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import Trainer, TrainerCallback

from pptrain.core.base import Mechanism
from pptrain.core.config import RunConfig
from pptrain.core.plotting import save_training_summary_plot
from pptrain.core.transfer import TransferBundle
from pptrain.integrations.base import CausalLMAdapter


@dataclass(slots=True)
class PrePreTrainingRun:
    run_dir: Path
    model_dir: Path
    metrics: dict[str, Any]
    plot_path: Path | None = None

    def load_transfer_bundle(self) -> TransferBundle:
        return TransferBundle.load(self.run_dir)


class EpochTrainDatasetRefreshCallback(TrainerCallback):
    def __init__(
        self,
        *,
        mechanism: Mechanism,
        train_dataset: Any,
        seed: int | None,
    ) -> None:
        self.mechanism = mechanism
        self.train_dataset = train_dataset
        self.seed = seed
        self.history: list[dict[str, Any]] = []

    def on_epoch_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        epoch_index = int(state.epoch or 0)
        if epoch_index <= 0:
            return control
        metadata = self.mechanism.refresh_train_dataset(
            self.train_dataset,
            seed=self.seed,
            epoch_index=epoch_index,
        )
        if metadata is not None:
            self.history.append(dict(metadata))
        return control


class PrePreTrainer:
    def __init__(
        self,
        mechanism: Mechanism,
        model_adapter: CausalLMAdapter,
        run_config: RunConfig,
    ) -> None:
        self.mechanism = mechanism
        self.model_adapter = model_adapter
        self.run_config = run_config

    def fit(self) -> PrePreTrainingRun:
        datasets = self.mechanism.build_datasets(seed=self.run_config.seed)
        tokenizer_spec = self.mechanism.tokenizer_spec()
        model = self.model_adapter.create_prepretrain_model(tokenizer_spec)
        refresh_callback = None
        if self.mechanism.uses_epoch_train_dataset_refresh():
            refresh_callback = EpochTrainDatasetRefreshCallback(
                mechanism=self.mechanism,
                train_dataset=datasets.train_dataset,
                seed=self.run_config.seed,
            )

        run_dir = Path(self.run_config.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(
            model=model,
            args=self.run_config.to_training_arguments(has_eval=datasets.eval_dataset is not None),
            train_dataset=datasets.train_dataset,
            eval_dataset=datasets.eval_dataset,
            data_collator=datasets.data_collator,
            callbacks=[refresh_callback] if refresh_callback is not None else None,
        )

        train_result = trainer.train()
        metrics = dict(train_result.metrics)
        if datasets.eval_dataset is not None:
            metrics.update(trainer.evaluate())
        if refresh_callback is not None and refresh_callback.history:
            datasets.metadata["train_refresh_history"] = list(refresh_callback.history)
            metrics["train_dataset_refresh_count"] = len(refresh_callback.history)

        model_dir = run_dir / "prepretrained_model"
        trainer.save_model(str(model_dir))
        trainer.save_state()
        plot_path = save_training_summary_plot(
            log_history=trainer.state.log_history,
            metrics=metrics,
            dataset_metadata=datasets.metadata,
            output_path=run_dir / "training_summary.png",
        )
        self._save_metadata(run_dir, tokenizer_spec.to_dict(), datasets.metadata, metrics)

        bundle = TransferBundle(
            run_dir=run_dir,
            model_dir=model_dir,
            tokenizer_spec=tokenizer_spec.to_dict(),
            mechanism_name=self.mechanism.name,
            mechanism_config=self.mechanism.export_config(),
            transfer_policy_name=self.mechanism.default_transfer_policy_name(),
        )
        bundle.save()
        return PrePreTrainingRun(
            run_dir=run_dir,
            model_dir=model_dir,
            metrics=metrics,
            plot_path=plot_path,
        )

    def _save_metadata(
        self,
        run_dir: Path,
        tokenizer_spec: dict[str, Any],
        dataset_metadata: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        payload = {
            "mechanism": {
                "name": self.mechanism.name,
                "config": self.mechanism.export_config(),
            },
            "model": self.model_adapter.config.to_dict(),
            "run": asdict(self.run_config),
            "tokenizer_spec": tokenizer_spec,
            "dataset_metadata": dataset_metadata,
            "metrics": metrics,
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
