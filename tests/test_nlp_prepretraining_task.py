from __future__ import annotations

from pptrain.core.registry import create_task, registered_tasks
from pptrain.tasks import nlp_prepretraining


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 1
    bos_token_id = 2
    eos_token = "<eos>"
    pad_token = None

    def __len__(self) -> int:
        return 32

    def __call__(self, text: str, *, add_special_tokens: bool = False, verbose: bool = False):
        del add_special_tokens, verbose
        ids = [3 + (sum(ord(char) for char in token) % 20) for token in text.split()]
        return {"input_ids": ids}


def test_nlp_prepretraining_task_builds_inline_bundle(monkeypatch) -> None:
    monkeypatch.setattr(
        nlp_prepretraining.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    task = create_task(
        "nlp_prepretraining",
        {
            "tokenizer_name_or_path": "fake-tokenizer",
            "source": "inline",
            "inline_train_texts": (
                "alpha beta gamma delta epsilon zeta eta theta",
                "iota kappa lambda mu nu xi omicron pi",
            ),
            "inline_eval_texts": ("rho sigma tau upsilon phi chi psi omega",),
            "max_length": 4,
        },
    )

    spec = task.tokenizer_spec()
    bundle = task.build_datasets(seed=13)

    assert spec.vocab_size == 32
    assert spec.pad_token_id == 1
    assert bundle.eval_dataset is not None
    assert len(bundle.train_dataset) >= 1
    assert len(bundle.eval_dataset) >= 1
    assert bundle.metadata["task"] == "nlp_prepretraining"
    assert bundle.metadata["train_sequence_count"] == len(bundle.train_dataset)


def test_nlp_prepretraining_task_is_registered() -> None:
    assert "nlp_prepretraining" in {task.name for task in registered_tasks()}
