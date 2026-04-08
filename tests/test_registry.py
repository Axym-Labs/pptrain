from pptrain.core.registry import create_mechanism, registered_mechanisms
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
