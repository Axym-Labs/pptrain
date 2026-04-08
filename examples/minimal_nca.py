from pptrain import PrePreTrainer, RunConfig, create_mechanism
from pptrain.integrations import HFCausalLMAdapter, HFModelConfig


def main() -> None:
    trainer = PrePreTrainer(
        mechanism=create_mechanism(
            "nca",
            {
                "preset": "smoke",
                "init_rollout_steps": 4,
            },
        ),
        model_adapter=HFCausalLMAdapter(
            HFModelConfig(
                model_name_or_path="sshleifer/tiny-gpt2",
                config_overrides={"n_positions": 256},
            )
        ),
        run_config=RunConfig(
            output_dir="runs/example-nca",
            max_steps=20,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            logging_steps=5,
            save_steps=20,
            eval_steps=20,
        ),
    )
    run = trainer.fit()
    print(run.run_dir)


if __name__ == "__main__":
    main()
