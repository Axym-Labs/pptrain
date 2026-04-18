from __future__ import annotations

from pptrain.core.base import TokenizerSpec
from pptrain.core.registry import create_task
from pptrain.reference_parity import (
    ReferenceFixture,
    ReferenceSplit,
    assert_task_fixture_matches,
    fixture_from_task,
    load_reference_fixture,
    save_reference_fixture,
)


def test_reference_fixture_round_trips_normalized_examples(tmp_path) -> None:
    fixture = ReferenceFixture(
        task_name="procedural",
        comparison_target="normalized_examples",
        tokenizer_spec=TokenizerSpec(vocab_size=8, pad_token_id=0),
        train=ReferenceSplit(examples=[{"task": "copy", "source": "abba", "target": "abba"}]),
        eval=ReferenceSplit(examples=[{"task": "reverse", "source": "abba", "target": "abba"}]),
    )

    path = tmp_path / "normalized_fixture.json"
    save_reference_fixture(fixture, path)
    reloaded = load_reference_fixture(path)

    assert reloaded == fixture


def test_fixture_matcher_accepts_normalized_examples_for_procedural_tasks() -> None:
    task = create_task(
        "procedural",
        {
            "preset": "smoke",
            "tasks": ("copy",),
            "sequence_count": 4,
            "eval_sequence_count": 2,
        },
    )

    fixture = fixture_from_task(
        task,
        preset_name="smoke",
        seed=11,
        comparison_target="normalized_examples",
    )

    assert fixture.train.examples is not None
    assert_task_fixture_matches(task, fixture)


def test_fixture_matcher_accepts_normalized_examples_for_lime_tasks() -> None:
    task = create_task(
        "lime",
        {
            "preset": "smoke",
            "modes": ("induct",),
            "sequence_count": 3,
            "eval_sequence_count": 2,
        },
    )

    fixture = fixture_from_task(
        task,
        preset_name="smoke",
        seed=13,
        comparison_target="normalized_examples",
    )

    assert fixture.train.examples is not None
    assert_task_fixture_matches(task, fixture)


def test_fixture_matcher_accepts_normalized_examples_for_summarization_tasks() -> None:
    task = create_task(
        "summarization",
        {
            "preset": "smoke",
            "tasks": ("copy_bulleted",),
            "sequence_count": 3,
            "eval_sequence_count": 2,
        },
    )

    fixture = fixture_from_task(
        task,
        preset_name="smoke",
        seed=17,
        comparison_target="normalized_examples",
    )

    assert fixture.train.examples is not None
    assert_task_fixture_matches(task, fixture)
