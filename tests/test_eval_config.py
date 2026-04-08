from pptrain.eval.config import build_eval_harness


def test_build_eval_harness_from_config() -> None:
    harness = build_eval_harness(
        {
            "tasks": [
                {
                    "type": "perplexity",
                    "dataset_name": "wikitext",
                    "dataset_config_name": "wikitext-2-raw-v1",
                    "split": "validation[:8]",
                },
                {
                    "type": "gsm8k",
                    "split": "test[:4]",
                },
            ]
        }
    )

    names = [task.name for task in harness.tasks]
    assert names == ["perplexity", "gsm8k"]
