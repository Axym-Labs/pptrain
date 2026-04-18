# Quickstart

This example uses a paper-backed preset, trains a small upstream model, and exports a transfer bundle for downstream pretraining.

```python
from pptrain import PrePreTrainer, RunConfig, create_task
from pptrain.integrations import HFCausalLMAdapter, HFModelConfig

task = create_task(
    "simpler_tasks",
    {
        "preset": "paper_binary_1m",
        "sequence_count": 256,
        "eval_sequence_count": 64,
        "max_length": 128,
    },
)

trainer = PrePreTrainer(
    task=task,
    model_adapter=HFCausalLMAdapter(
        HFModelConfig(
            model_name_or_path="sshleifer/tiny-gpt2",
            config_overrides={"n_positions": 128},
        )
    ),
    run_config=RunConfig(
        output_dir="runs/smoke",
        max_steps=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=5,
        save_steps=20,
        eval_steps=20,
    ),
)

run = trainer.fit()
bundle = run.load_transfer_bundle()
```

If you want to evaluate the transferred checkpoint immediately afterward, use:

```bash
pptrain fit configs/nca_minimal.yaml --eval-config configs/eval_perplexity_smoke.yaml
```
