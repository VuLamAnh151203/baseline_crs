import json
from pathlib import Path

import pytest
import torch

from tdrl.diffusion_model import ScoreTransformer, load_initial_policy
from tdrl.provenance import file_hash

FIXTURE_PATH = Path(__file__).parent / "fixtures/difftod-tiny-native-model.pt"
META = json.loads(FIXTURE_PATH.with_suffix(".json").read_text())


def model_fixture():
    assert file_hash(FIXTURE_PATH) == META["fixture_sha256"]
    fixture = torch.load(FIXTURE_PATH, weights_only=True)
    model = ScoreTransformer(fixture["config"]).eval()
    model.load_state_dict(fixture["weights"], strict=True)
    return model, fixture


def test_native_ddit_checkpoint_keys_forward_and_gradients():
    model, fixture = model_fixture()
    output = model(fixture["tokens"], fixture["sigma"])
    torch.testing.assert_close(output, fixture["scores"], **META["forward_tolerance"])
    output.sum().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, fixture["gradients"][name], **META["gradient_tolerance"]
        )
    # Native self-token log score is fixed to zero, not an unconstrained LM logit.
    assert not output.gather(-1, fixture["tokens"][..., None]).any()
    assert (model.blocks[0].adaLN_modulation.weight != 0).any()


def test_publishing_bf16_parameters_preserves_full_precision_rope_frequencies():
    model, fixture = model_fixture()
    frequencies = model.rotary_emb.inv_freq.clone()
    model.to(dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert model.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.equal(frequencies, model.rotary_emb.inv_freq)


def test_meta_initialization_assigns_native_weights_before_materialization():
    _, fixture = model_fixture()
    with torch.device("meta"):
        model = ScoreTransformer(fixture["config"])
    model.load_state_dict(fixture["weights"], strict=True, assign=True)
    model.to(dtype=torch.bfloat16)
    assert all(p.device.type == "cpu" and p.dtype == torch.bfloat16 for p in model.parameters())


def test_invalid_sigma_and_unknown_diffusion_graph_fail_closed():
    model, fixture = model_fixture()
    with pytest.raises(ValueError, match="sigma"):
        model(fixture["tokens"], torch.zeros(2))
    config = {**fixture["config"], "graph": {"type": "uniform"}}
    with pytest.raises(ValueError, match="absorbing"):
        ScoreTransformer(config)


def test_pretrained_policy_loader_rejects_non_slurm_before_touching_weights(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        load_initial_policy(tmp_path)
