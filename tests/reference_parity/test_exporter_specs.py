from __future__ import annotations

from pptrain.reference_parity import REFERENCE_EXPORTER_SPECS


def test_reference_exporter_specs_cover_core_open_source_tasks() -> None:
    expected = {"lime", "summarization", "procedural", "nca"}
    expected_targets = {
        "lime": "normalized_examples",
        "summarization": "normalized_examples",
        "procedural": "normalized_examples",
        "nca": "dataset_bundle",
    }
    assert expected.issubset(REFERENCE_EXPORTER_SPECS)

    for task_name in expected:
        spec = REFERENCE_EXPORTER_SPECS[task_name]
        assert spec.task_name == task_name
        assert spec.fixture_format_version == 1
        assert spec.reference_repo.startswith("https://github.com/")
        assert spec.comparison_target == expected_targets[task_name]
        assert spec.expected_splits == ("train", "eval")
        assert spec.recommended_presets
        assert spec.generator_hint
