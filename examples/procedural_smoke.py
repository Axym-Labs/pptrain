from pptrain import PrePreTrainer, RunConfig
from pptrain.integrations import HFCausalLMAdapter, HFModelConfig
from pptrain.mechanisms.procedural import ProceduralConfig, ProceduralMechanism


def main() -> None:
    trainer = PrePreTrainer(
        mechanism=ProceduralMechanism(
            ProceduralConfig(
                tasks=("copy", "reverse", "sort", "addition"),
                sequence_count=256,
                eval_sequence_count=64,
                max_length=96,
            )
        ),
        model_adapter=HFCausalLMAdapter(
            HFModelConfig(
                model_name_or_path="sshleifer/tiny-gpt2",
                config_overrides={"n_positions": 96},
            )
        ),
        run_config=RunConfig(
            output_dir="runs/example-procedural",
            max_steps=20,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_steps=5,
            save_steps=20,
            eval_steps=20,
        ),
    )
    run = trainer.fit()
    print(run.run_dir)
    print(run.plot_path)


if __name__ == "__main__":
    main()

