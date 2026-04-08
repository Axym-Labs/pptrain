from pptrain.eval import ARCAGI2Dataset, score_arc_predictions


def main() -> None:
    dataset = ARCAGI2Dataset.from_directory("path/to/ARC-AGI-2/data/training")
    trivial_predictions = {
        task.task_id: [pair.input for pair in task.test]
        for task in dataset.tasks[:10]
    }
    score = score_arc_predictions(
        dataset=ARCAGI2Dataset(tasks=dataset.tasks[:10]),
        predictions=trivial_predictions,
    )
    print(score)


if __name__ == "__main__":
    main()

