from pptrain.eval import ARCAGI2Task, EvalHarness


def echo_input_predictor(task):
    return [pair.input for pair in task.test]


def main() -> None:
    harness = EvalHarness(
        [
            ARCAGI2Task(
                data_dir="path/to/ARC-AGI-2/data/training",
                max_tasks=10,
            )
        ]
    )
    results = harness.run_and_save(
        output_dir="artifacts/arc-eval",
        predictor=echo_input_predictor,
    )
    print(results["arc_agi2"].metrics)


if __name__ == "__main__":
    main()
