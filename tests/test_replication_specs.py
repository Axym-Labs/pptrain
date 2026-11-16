from pptrain.integrations.hf import HFModelConfig
from pptrain.replication.runner import (
    _apply_run_config_overrides,
    _natural_warmup_run_config,
    _nlp_prepretraining_task_config,
    _perplexity_from_metrics,
    matched_natural_block_size,
)
from pptrain.replication.specs import build_replication_profile


def test_smoke_replication_profile_contains_all_tasks(tmp_path) -> None:
    profile = build_replication_profile("smoke", output_dir=str(tmp_path), test_mode=True)
    names = [study.task_name for study in profile.studies]
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
    assert profile.synthetic_run_config.max_steps == 72
    assert profile.synthetic_run_config.learning_rate == 1e-5
    assert profile.synthetic_run_config.save_steps == 18
    assert profile.synthetic_run_config.warmup_steps == 4
    assert profile.downstream_run_config.max_steps == 1764
    assert profile.downstream_run_config.learning_rate == 1e-6
    assert profile.downstream_run_config.save_steps == 441
    assert profile.downstream_run_config.warmup_steps == 88
    simpler_tasks = next(study for study in profile.studies if study.task_name == "simpler_tasks")
    assert simpler_tasks.synthetic_run_config_overrides["learning_rate"] == 1e-6
    assert simpler_tasks.synthetic_run_config_overrides["max_steps"] == 288
    assert simpler_tasks.synthetic_run_config_overrides["warmup_steps"] == 14
    assert simpler_tasks.downstream_run_config_overrides == {"learning_rate": 1e-6}
    assert profile.synthetic_run_config.remove_checkpoints is True
    assert profile.downstream_run_config.remove_checkpoints is True
    assert profile.seed_values == (11, 23, 37)
    assert profile.datasets["general_text"].streaming is True
    assert profile.datasets["general_text"].dataset_name == "HuggingFaceFW/fineweb-edu"
    assert profile.datasets["math_text"].dataset_name == "HuggingFaceTB/finemath"
    assert profile.datasets["summary_text"].dataset_name == "vblagoje/cc_news"
    nca = next(study for study in profile.studies if study.task_name == "nca")
    assert nca.synthetic_run_config_overrides["max_steps"] == 72
    lime = next(study for study in profile.studies if study.task_name == "lime")
    assert lime.synthetic_run_config_overrides["max_steps"] == 288
    summarization = next(study for study in profile.studies if study.task_name == "summarization")
    assert summarization.synthetic_run_config_overrides["max_steps"] == 288


def test_wikitext_proxy_profile_uses_downloadable_hf_slices(tmp_path) -> None:
    profile = build_replication_profile("wikitext_proxy", output_dir=str(tmp_path), test_mode=False)

    assert profile.context_length == 512
    assert profile.model_name_or_path == "EleutherAI/pythia-70m-deduped"
    assert profile.datasets["general_text"].dataset_name == "Salesforce/wikitext"
    assert profile.datasets["general_text"].dataset_config_name == "wikitext-2-raw-v1"
    assert profile.datasets["general_text"].streaming is False
    assert profile.datasets["general_text"].warmup_split == "train[:512]"
    assert profile.datasets["general_text"].train_split == "train[512:2048]"
    assert profile.downstream_run_config.max_steps == 256
    assert all(study.compare_against_natural_warmup for study in profile.studies)


def test_perplexity_from_metrics_saturates_on_large_eval_loss() -> None:
    assert _perplexity_from_metrics({"eval_loss": 1_000.0}) == float("inf")


def test_nlp_prepretraining_arm_matches_each_synthetic_upstream_budget(tmp_path) -> None:
    profile = build_replication_profile(
        "paper_proxy_2048", output_dir=str(tmp_path), test_mode=False, context_length=12288
    )
    for study in profile.studies:
        synthetic = _apply_run_config_overrides(profile.synthetic_run_config, study.synthetic_run_config_overrides)
        natural = _natural_warmup_run_config(study=study, profile=profile)
        for field_name in (
            "max_steps",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "warmup_steps",
        ):
            assert getattr(natural, field_name) == getattr(synthetic, field_name), (study.task_name, field_name)

    # Text blocks as long as the mean synthetic sequence give equal upstream token counts.
    assert matched_natural_block_size(synthetic_mean_sequence_tokens=102.24, context_length=12288) == 102
    assert matched_natural_block_size(synthetic_mean_sequence_tokens=12288.0, context_length=12288) == 12288
    assert matched_natural_block_size(synthetic_mean_sequence_tokens=20000.0, context_length=12288) == 12288
    lime = next(study for study in profile.studies if study.task_name == "lime")
    config = _nlp_prepretraining_task_config(
        study=lime,
        profile=profile,
        hf_config=HFModelConfig(model_name_or_path=profile.model_name_or_path),
        run_config=_natural_warmup_run_config(study=lime, profile=profile),
        block_size=102,
    )
    assert config["max_length"] == 102
    assert config["train_target_token_count"] >= 288 * 8 * 102

