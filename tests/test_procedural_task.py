import numpy as np

from pptrain.core.base import SymbolicTask
from pptrain.tasks.procedural import ProceduralProgram
from pptrain.tasks import ProceduralConfig, ProceduralTaskFamily


def test_procedural_task_builds_sequences() -> None:
    task = ProceduralTaskFamily(
        ProceduralConfig(
            tasks=("identity", "reverse", "sort", "set", "union", "delete", "addition"),
            sequence_count=8,
            eval_sequence_count=3,
            max_length=64,
            max_symbol_length=10,
            max_number=99,
        )
    )
    bundle = task.build_datasets(seed=11)
    assert len(bundle.train_dataset) == 8
    assert len(bundle.eval_dataset) == 3
    sample = bundle.train_dataset[0]
    assert sample["input_ids"].shape[0] <= 64
    assert sample["labels"].shape[0] <= 64


def test_procedural_delete_removes_one_requested_symbol_once() -> None:
    task = ProceduralTaskFamily(ProceduralConfig(tasks=("delete",)))

    executed = task.execute_task(
        SymbolicTask(
            name="delete",
            payload=ProceduralProgram(left="abac", right="a"),
        )
    )

    assert executed.payload == "delete:abac|a=>bac"


def test_procedural_delete_samples_single_symbol_query() -> None:
    task = ProceduralTaskFamily(ProceduralConfig(tasks=("delete",), min_symbol_length=4, max_symbol_length=4))

    program = task.sample_task(np.random.default_rng(7)).payload

    assert isinstance(program.right, str)
    assert len(program.right) == 1
