from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from transformers import AutoTokenizer

from pptrain.core.base import DatasetBundle, Task, TokenizerSpec
from pptrain.core.registry import register_task
from pptrain.replication.data import build_text_sequence_bundle, build_text_train_eval_bundle
from pptrain.replication.specs import TextDatasetSpec


@dataclass(frozen=True, slots=True)
class NLPPrepretrainingConfig:
    tokenizer_name_or_path: str
    source: str = "hf"
    formatter: str = "plain_text"
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    subset: str | None = None
    train_split: str | None = "train"
    eval_split: str | None = None
    text_field: str = "text"
    streaming: bool = False
    shuffle_buffer_size: int | None = None
    shuffle_seed: int = 13
    train_skip_records: int = 0
    eval_skip_records: int = 0
    inline_train_texts: tuple[str, ...] = ()
    inline_eval_texts: tuple[str, ...] = ()
    max_length: int = 512
    train_target_token_count: int | None = None
    eval_target_token_count: int | None = None
    trust_remote_code: bool = False


class NLPPrepretrainingTask(Task):
    name = "nlp_prepretraining"

    def __init__(self, config: NLPPrepretrainingConfig) -> None:
        super().__init__(config)
        self._tokenizer = None

    def tokenizer_spec(self) -> TokenizerSpec:
        tokenizer = self._load_tokenizer()
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        return TokenizerSpec(
            vocab_size=len(tokenizer),
            pad_token_id=int(pad_token_id),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    def build_datasets(self, seed: int | None = None) -> DatasetBundle:
        del seed
        dataset_spec = self._dataset_spec()
        tokenizer = self._load_tokenizer()
        has_eval = self.config.eval_split is not None or bool(self.config.inline_eval_texts)
        if has_eval:
            bundle = build_text_train_eval_bundle(
                tokenizer=tokenizer,
                dataset_spec=dataset_spec,
                block_size=self.config.max_length,
                train_target_token_count=self.config.train_target_token_count,
                eval_target_token_count=self.config.eval_target_token_count,
            )
        else:
            text_bundle = build_text_sequence_bundle(
                tokenizer=tokenizer,
                dataset_spec=dataset_spec,
                block_size=self.config.max_length,
                split="train",
                target_token_count=self.config.train_target_token_count,
            )
            bundle = text_bundle.dataset_bundle
        bundle.metadata.update(
            {
                "task": self.name,
                "config": self.export_config(),
                "train_sequence_count": len(bundle.train_dataset),
                "eval_sequence_count": len(bundle.eval_dataset) if bundle.eval_dataset is not None else 0,
            }
        )
        return bundle

    def export_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def _load_tokenizer(self):
        if self._tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer = tokenizer
        return self._tokenizer

    def _dataset_spec(self) -> TextDatasetSpec:
        return TextDatasetSpec(
            source=self.config.source,
            formatter=self.config.formatter,
            dataset_name=self.config.dataset_name,
            dataset_config_name=self.config.dataset_config_name,
            subset=self.config.subset,
            warmup_split=self.config.train_split,
            train_split=self.config.train_split,
            eval_split=self.config.eval_split,
            text_field=self.config.text_field,
            streaming=self.config.streaming,
            shuffle_buffer_size=self.config.shuffle_buffer_size,
            shuffle_seed=self.config.shuffle_seed,
            warmup_skip_records=self.config.train_skip_records,
            train_skip_records=self.config.train_skip_records,
            eval_skip_records=self.config.eval_skip_records,
            inline_warmup_texts=self.config.inline_train_texts,
            inline_train_texts=self.config.inline_train_texts,
            inline_eval_texts=self.config.inline_eval_texts,
        )


def _coerce_config(config: dict[str, Any]) -> NLPPrepretrainingConfig:
    payload = dict(config)
    for key in ("inline_train_texts", "inline_eval_texts"):
        if key in payload:
            payload[key] = tuple(payload[key])
    return NLPPrepretrainingConfig(**payload)


register_task(
    "nlp_prepretraining",
    lambda config: NLPPrepretrainingTask(_coerce_config(config)),
    description="Natural-language Hugging Face or inline text pre-pretraining task.",
)
