from __future__ import annotations

from dataclasses import replace

import pytest

from pptrain.core.base import TokenizerSpec
from pptrain.core.registry import create_task
from pptrain.reference_parity import (
    ReferenceFixture,
    ReferenceFixtureMismatch,
    ReferenceSource,
    ReferenceSplit,
    assert_task_fixture_matches,
    fixture_from_task,
    load_reference_fixture,
    save_reference_fixture,
)


def test_reference_fixture_round_trips_through_json(tmp_path) -> None:
    fixture = ReferenceFixture(
        task_name="lime",
        preset_name="paper_benchmark_100k",
        seed=7,
        tokenizer_spec=TokenizerSpec(vocab_size=32, pad_token_id=0, bos_token_id=1, eos_token_id=2),
        train=ReferenceSplit(
            sequences=[[1, 4, 5, 2], [1, 8, 9, 2]],
            labels=[[1, 4, 5, 2], [1, 8, 9, 2]],
            metadata={"task_counts": {"induct": 2}},
        ),
        eval=ReferenceSplit(
            sequences=[[1, 3, 7, 2]],
            labels=[[1, 3, 7, 2]],
            metadata={"task_counts": {"deduct": 1}},
        ),
        metadata={"train_sequence_count": 2, "eval_sequence_count": 1},
        source=ReferenceSource(
            repo="https://github.com/tonywu95/LIME",
            commit="abc1234",
            generator="reason/generate_data.py",
        ),
    )

    path = tmp_path / "fixture.json"
    save_reference_fixture(fixture, path)
    reloaded = load_reference_fixture(path)

    assert reloaded == fixture


def test_fixture_matcher_accepts_identical_task_outputs() -> None:
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
        source=ReferenceSource(
            repo="https://github.com/zlshinnick/procedural-pretraining",
            commit="deadbeef",
            generator="procedural_data",
        ),
    )

    assert_task_fixture_matches(task, fixture)


def test_fixture_matcher_reports_sequence_mismatch() -> None:
    task = create_task(
        "procedural",
        {
            "preset": "smoke",
            "tasks": ("copy",),
            "sequence_count": 4,
            "eval_sequence_count": 2,
        },
    )
    fixture = fixture_from_task(task, preset_name="smoke", seed=11)
    broken_train = replace(
        fixture.train,
        sequences=[[999, *fixture.train.sequences[0][1:]], *fixture.train.sequences[1:]],
    )
    broken_fixture = replace(fixture, train=broken_train)

    with pytest.raises(ReferenceFixtureMismatch, match="train sequences differ at index 0"):
        assert_task_fixture_matches(task, broken_fixture)


def test_load_reference_fixture_rejects_missing_required_fields(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"task_name":"lime"}', encoding="utf-8")

    with pytest.raises(ValueError, match="format_version"):
        load_reference_fixture(path)
