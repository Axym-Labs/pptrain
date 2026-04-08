from pptrain.replication.specs import build_replication_profile


def test_smoke_replication_profile_contains_all_mechanisms(tmp_path) -> None:
    profile = build_replication_profile("smoke", output_dir=str(tmp_path), test_mode=True)
    names = [study.mechanism_name for study in profile.studies]
    assert names == ["nca", "lime", "simpler_tasks", "procedural", "dyck", "summarization"]
    assert profile.context_length == 128


def test_full_replication_profile_defaults_to_2048_context(tmp_path) -> None:
    profile = build_replication_profile("paper_proxy_2048", output_dir=str(tmp_path), test_mode=False)
    assert profile.context_length == 2048
    assert profile.model_name_or_path == "EleutherAI/pythia-160m-deduped"
