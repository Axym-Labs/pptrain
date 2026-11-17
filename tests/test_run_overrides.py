import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pptrain.cli import build_parser
from pptrain.core.config import RunConfig
from pptrain.core.transfer import ReinitializeEmbeddingTransferPolicy
from pptrain.replication.runner import apply_profile_overrides
from pptrain.replication.specs import build_replication_profile


def test_run_config_passes_optimizer_and_liger(tmp_path) -> None:
    args = RunConfig(
        output_dir=str(tmp_path),
        optim="adafactor",
        optim_args="beta1=None",
        use_liger_kernel=True,
    ).to_training_arguments(has_eval=False)
    assert args.optim == "adafactor"
    assert args.optim_args == "beta1=None"
    assert args.use_liger_kernel is True
    default = RunConfig(output_dir=str(tmp_path)).to_training_arguments(has_eval=False)
    assert default.optim == "adamw_torch"
    assert default.use_liger_kernel is False


def test_profile_overrides_reach_every_stage(tmp_path) -> None:
    profile = build_replication_profile("paper_proxy_2048", output_dir=str(tmp_path), test_mode=False)
    updated = apply_profile_overrides(
        profile,
        config_overrides={"num_hidden_layers": 24},
        synthetic_run_overrides={"gradient_checkpointing": False},
        downstream_run_overrides={"per_device_train_batch_size": 1, "gradient_accumulation_steps": 8},
    )
    assert updated.config_overrides["num_hidden_layers"] == 24
    assert updated.synthetic_run_config.gradient_checkpointing is False
    assert updated.downstream_run_config.per_device_train_batch_size == 1
    assert updated.downstream_run_config.gradient_accumulation_steps == 8
    # untouched fields and the study-level overrides stay as they were
    assert updated.downstream_run_config.learning_rate == profile.downstream_run_config.learning_rate
    assert updated.studies == profile.studies
    assert apply_profile_overrides(profile) == profile


def test_cli_parses_json_overrides() -> None:
    args = build_parser().parse_args(
        [
            "replicate",
            "--downstream-run-overrides",
            json.dumps({"optim": "adafactor"}),
            "--config-overrides",
            '{"num_hidden_layers": 24}',
        ]
    )
    assert args.downstream_run_overrides == {"optim": "adafactor"}
    assert args.config_overrides == {"num_hidden_layers": 24}
    assert args.synthetic_run_overrides is None


def _tiny_qwen3(vocab_size: int) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


@pytest.mark.parametrize("target_vocab", [300, 20])
def test_qwen3_transfer_with_tied_embeddings(target_vocab: int) -> None:
    source = _tiny_qwen3(20)
    target = _tiny_qwen3(target_vocab)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(1.0)
    original_embedding = target.get_input_embeddings().weight.detach().clone()
    report = ReinitializeEmbeddingTransferPolicy().apply_state_dict(source.state_dict(), target)
    assert torch.equal(target.get_input_embeddings().weight, original_embedding)
    assert target.get_output_embeddings().weight is target.get_input_embeddings().weight
    source_state = source.state_dict()
    for name, tensor in target.state_dict().items():
        if "embed_tokens" in name or name == "lm_head.weight":
            continue
        assert torch.equal(tensor, source_state[name]), name
    assert report.loaded_parameter_count > 0


def test_pruning_keeps_shared_scratch(tmp_path) -> None:
    from pptrain.replication.runner import SHARED_SCRATCH_DIRNAME, _prune_seed_artifacts

    shared = tmp_path / SHARED_SCRATCH_DIRNAME / "general_text_abc" / "seed_11"
    own = tmp_path / "nca" / "seed_11" / "nlp_prepretraining" / "downstream"
    for path in (shared / "model", own / "model"):
        path.mkdir(parents=True)
        (path / "model.safetensors").write_text("x")
    _prune_seed_artifacts(
        {
            "variants": {
                "scratch": {"model_dir": str(shared / "model"), "run_dir": str(shared)},
                "nlp_prepretraining": {"model_dir": str(own / "model"), "run_dir": str(own)},
            }
        }
    )
    assert (shared / "model" / "model.safetensors").exists()
    assert not own.exists()


def test_layer_count_override_survives_save_and_reload(tmp_path) -> None:
    from transformers import AutoModelForCausalLM

    from pptrain.integrations.hf import apply_config_overrides, causal_lm_from_config

    config = Qwen3Config(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    apply_config_overrides(config, {"num_hidden_layers": 2})
    assert config.layer_types == ["full_attention"] * 2
    model = causal_lm_from_config(config)
    model.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert reloaded.config.num_hidden_layers == 2
