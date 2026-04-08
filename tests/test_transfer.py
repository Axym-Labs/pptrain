import torch
from transformers import GPT2Config, GPT2LMHeadModel

from pptrain.core.transfer import ReinitializeEmbeddingTransferPolicy, SkipParametersTransferPolicy
from pptrain.integrations import CallableCausalLMAdapter


def test_transfer_skips_embeddings_and_loads_matching_weights() -> None:
    source = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=32))
    target = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=64))
    source_state = source.state_dict()

    original_embedding = target.get_input_embeddings().weight.detach().clone()
    policy = ReinitializeEmbeddingTransferPolicy()
    report = policy.apply_state_dict(source_state, target)

    assert report.loaded_parameter_count > 0
    assert torch.equal(target.get_input_embeddings().weight, original_embedding)
    assert any("wte" in name or "lm_head" in name for name in report.skipped_parameters)


def test_skip_parameters_transfer_policy_skips_named_prefixes() -> None:
    source = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=32))
    target = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=32))
    original_embedding = target.get_input_embeddings().weight.detach().clone()

    report = SkipParametersTransferPolicy(
        skip_parameter_prefixes=("transformer.wte", "lm_head")
    ).apply_state_dict(
        source.state_dict(),
        target,
    )

    assert report.loaded_parameter_count > 0
    assert torch.equal(target.get_input_embeddings().weight, original_embedding)


def test_callable_adapter_wraps_custom_construction() -> None:
    adapter = CallableCausalLMAdapter(
        create_prepretrain_model=lambda tokenizer_spec: GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=tokenizer_spec.vocab_size)
        ),
        load_downstream_model=lambda: GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16)
        ),
        name="custom-gpt2",
    )

    prepretrain_model = adapter.create_prepretrain_model(
        type("TokenizerSpecLike", (), {"vocab_size": 32})()
    )
    downstream_model = adapter.load_downstream_model()

    assert prepretrain_model.config.vocab_size == 32
    assert downstream_model.config.vocab_size == 16
    assert adapter.config.to_dict()["name"] == "custom-gpt2"
