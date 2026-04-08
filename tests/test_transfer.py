import torch
from transformers import GPT2Config, GPT2LMHeadModel

from pptrain.core.transfer import ReinitializeEmbeddingTransferPolicy


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

