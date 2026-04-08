from __future__ import annotations

from transformers import GPT2Config, GPT2LMHeadModel

from pptrain import PrePreTrainer, RunConfig, create_mechanism
from pptrain.integrations import VocabSizeCausalLMAdapter
from pptrain.transfer import SkipParametersTransferPolicy


def build_prepretrain_model(vocab_size: int) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(
        GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=64,
            n_positions=128,
            vocab_size=vocab_size,
        )
    )


def load_downstream_model() -> GPT2LMHeadModel:
    return GPT2LMHeadModel(
        GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=64,
            n_positions=128,
            vocab_size=512,
        )
    )


def main() -> None:
    trainer = PrePreTrainer(
        mechanism=create_mechanism("simpler_tasks", {"preset": "smoke", "sequence_count": 64}),
        model_adapter=VocabSizeCausalLMAdapter(
            create_prepretrain_model=build_prepretrain_model,
            load_downstream_model=load_downstream_model,
            name="custom-gpt2",
        ),
        run_config=RunConfig(
            output_dir="runs/custom-adapter-example",
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
    target_model = load_downstream_model()
    report = SkipParametersTransferPolicy(
        skip_parameter_prefixes=("transformer.wte", "lm_head"),
    ).apply_bundle(bundle, target_model)
    print(report.loaded_parameter_count)


if __name__ == "__main__":
    main()
