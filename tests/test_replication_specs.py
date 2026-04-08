from pptrain.replication.specs import build_replication_profile


def test_smoke_replication_profile_contains_all_mechanisms(tmp_path) -> None:
    profile = build_replication_profile("smoke", output_dir=str(tmp_path), test_mode=True)
    names = [study.mechanism_name for study in profile.studies]
    assert names == ["nca", "lime", "simpler_tasks", "procedural", "dyck", "summarization"]
    assert profile.context_length == 128
    assert profile.seed_values == (41, 43, 47)
    assert profile.diagnostic_max_batches == 2
    for study in profile.studies:
        assert "compute_matched_gain" in study.claim_categories


def test_full_replication_profile_defaults_to_2048_context(tmp_path) -> None:
    profile = build_replication_profile("paper_proxy_2048", output_dir=str(tmp_path), test_mode=False)
    assert profile.context_length == 2048
    assert profile.model_name_or_path == "EleutherAI/pythia-160m-deduped"
    assert profile.seed_values == (11, 23, 37, 47, 59, 71, 83, 97, 109, 131)
    assert profile.datasets["general_text"].streaming is True
    assert profile.datasets["general_text"].dataset_name == "HuggingFaceFW/fineweb-edu"
    assert profile.datasets["math_text"].dataset_name == "HuggingFaceTB/finemath"
    assert profile.datasets["summary_text"].dataset_name == "vblagoje/cc_news"
