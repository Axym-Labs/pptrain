from pptrain.core.registry import create_task, registered_presets, registered_tasks
from pptrain.tasks import NCATask


def test_registry_builds_nca_task() -> None:
    task = create_task("nca", {"sequence_count": 4, "eval_sequence_count": 2})
    assert isinstance(task, NCATask)


def test_registry_lists_registered_tasks() -> None:
    names = [item.name for item in registered_tasks()]
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
    task = create_task(
        "nca",
        {
            "preset": "smoke",
            "sequence_count": 4,
            "eval_sequence_count": 2,
        },
    )
    assert isinstance(task, NCATask)
    assert task.config.grid_size == 8
    assert task.config.sequence_count == 4
    assert task.config.eval_sequence_count == 2
