from pptrain.core.registry import create_mechanism, registered_mechanisms, registered_presets
from pptrain.mechanisms import NCAMechanism


def test_registry_builds_nca_mechanism() -> None:
    mechanism = create_mechanism("nca", {"sequence_count": 4, "eval_sequence_count": 2})
    assert isinstance(mechanism, NCAMechanism)


def test_registry_lists_registered_mechanisms() -> None:
    names = [item.name for item in registered_mechanisms()]
    assert "dyck" in names
    assert "lime" in names
    assert "nca" in names
    assert "procedural" in names
    assert "simpler_tasks" in names
    assert "summarization" in names


def test_registry_exposes_presets() -> None:
    preset_names = [preset.name for preset in registered_presets("nca")]
    assert "paper_code" in preset_names
    assert "paper_web_text" in preset_names
    summarization_preset_names = [preset.name for preset in registered_presets("summarization")]
    assert "paper_ourtasks_subset_100k" in summarization_preset_names
    assert "paper_copy_quoted_100k" in summarization_preset_names


def test_registry_merges_preset_config_with_overrides() -> None:
    mechanism = create_mechanism(
        "nca",
        {
            "preset": "smoke",
            "sequence_count": 4,
            "eval_sequence_count": 2,
        },
    )
    assert isinstance(mechanism, NCAMechanism)
    assert mechanism.config.grid_size == 8
    assert mechanism.config.sequence_count == 4
    assert mechanism.config.eval_sequence_count == 2
