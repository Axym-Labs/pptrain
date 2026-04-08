from pptrain.core.registry import create_mechanism
from pptrain.mechanisms import NCAMechanism


def test_registry_builds_nca_mechanism() -> None:
    mechanism = create_mechanism("nca", {"sequence_count": 4, "eval_sequence_count": 2})
    assert isinstance(mechanism, NCAMechanism)

