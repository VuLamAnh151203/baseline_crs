#!/usr/bin/env python3
"""Pinned native DDiT architecture parity on a tiny CPU fixture, not inference."""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

from tdrl.diffusion_model import ScoreTransformer
from tdrl.provenance import file_hash


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def namespace(value):
    return types.SimpleNamespace(
        **{k: namespace(v) if isinstance(v, dict) else v for k, v in value.items()}
    )


def rearrange(tensor, pattern, **sizes):
    # Four upstream layout conversions; attention itself is an independent
    # explicit matrix-multiply/softmax reference on CPU, not flash emulation.
    if pattern == "b s (three h d) -> b s three h d":
        return tensor.reshape(*tensor.shape[:2], sizes["three"], sizes["h"], -1)
    if pattern == "b s ... -> (b s) ...":
        return tensor.reshape(-1, *tensor.shape[2:])
    if pattern == "(b s) h d -> b s (h d)":
        return tensor.reshape(sizes["b"], -1, tensor.shape[-2] * tensor.shape[-1])
    raise ValueError("unexpected native tensor layout: " + pattern)


def explicit_attention(qkv, cumulative_lengths, length, dropout, causal):
    assert dropout == 0 and not causal
    batches = len(cumulative_lengths) - 1
    query, key, value = qkv.reshape(batches, length, *qkv.shape[1:]).unbind(2)
    query, key, value = [x.transpose(1, 2) for x in [query, key, value]]
    weights = (query @ key.transpose(-1, -2) / query.shape[-1] ** 0.5).softmax(-1)
    return (weights @ value).transpose(1, 2).reshape(-1, *qkv.shape[-2:])


def audit(source, output, lock_path):
    lock = json.loads(lock_path.read_text())
    if (
        lock.get("schema_version") != 1
        or lock["revision"] != "fd46b2887a9c0d9e0e413623e79c72ee32b9b7c1"
    ):
        raise ValueError("unknown native transformer source contract")
    for name, expected in lock["files"].items():
        if file_hash(source / name) != expected:
            raise ValueError("native transformer hash mismatch: " + name)
    package = types.ModuleType("native_fixture")
    package.__path__ = [str(source / "model")]
    flash = types.ModuleType("flash_attn")
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.flash_attn_varlen_qkvpacked_func = explicit_attention
    hubs = types.ModuleType("huggingface_hub")
    hubs.PyTorchModelHubMixin = type("NoRemoteLoader", (), {})
    omega = types.ModuleType("omegaconf")
    omega.OmegaConf = types.SimpleNamespace(create=namespace)
    einops = types.ModuleType("einops")
    einops.rearrange = rearrange
    modules = {
        "native_fixture": package,
        "flash_attn": flash,
        "flash_attn.flash_attn_interface": interface,
        "huggingface_hub": hubs,
        "omegaconf": omega,
        "einops": einops,
    }
    with patch.dict(sys.modules, modules):
        package.rotary = load("native_fixture.rotary", source / "model/rotary.py")
        load("native_fixture.fused_add_dropout_scale", source / "model/fused_add_dropout_scale.py")
        native = load("native_fixture.transformer", source / "model/transformer.py")
        config = {
            "tokens": 7,
            "graph": {"type": "absorb"},
            "model": {
                "type": "ddit",
                "hidden_size": 16,
                "cond_dim": 8,
                "length": 8,
                "n_blocks": 2,
                "n_heads": 2,
                "scale_by_sigma": True,
                "dropout": 0.1,
            },
        }
        torch.manual_seed(17)
        reference = native.SEDD(config).eval()
        # Exercise attention and gated residuals rather than zero-initialized
        # modulation/output layers, which would conceal architectural mistakes.
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if "adaLN_modulation" in name or "output_layer.linear" in name:
                    parameter.normal_(0, 0.1)
        model = ScoreTransformer(config).eval()
        model.load_state_dict(reference.state_dict(), strict=True)
        tokens = torch.tensor([[7, 1, 2, 3, 7], [3, 7, 5, 6, 0]])
        sigma = torch.tensor([0.25, 0.9])
        expected = reference(tokens, sigma)
        actual = model(tokens, sigma)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        expected.sum().backward()
        actual.sum().backward()
        gradients = {}
        for name, parameter in reference.named_parameters():
            local = dict(model.named_parameters())[name]
            torch.testing.assert_close(local.grad, parameter.grad, rtol=2e-5, atol=2e-5)
            gradients[name] = parameter.grad
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": config,
                "weights": reference.state_dict(),
                "tokens": tokens,
                "sigma": sigma,
                "scores": expected.detach(),
                "gradients": gradients,
            },
            output / "difftod-tiny-native-model.pt",
        )
        record = {
            "schema_version": 1,
            "status": "passed",
            "scope": "tiny CPU architecture fixture only",
            "source": lock,
            "attention_reference": "explicit CPU matmul/softmax; native FlashAttention parity on H100 still required",
            "seed": 17,
            "forward_tolerance": {"rtol": 1e-6, "atol": 1e-6},
            "gradient_tolerance": {"rtol": 2e-5, "atol": 2e-5},
            "fixture_sha256": file_hash(output / "difftod-tiny-native-model.pt"),
            "pretrained_inference_performed": False,
        }
        (output / "difftod-tiny-native-model.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"status": "passed", "fixture_sha256": record["fixture_sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lock", type=Path, default=Path("configs/difftod-model-reference.lock.json")
    )
    args = parser.parse_args()
    audit(args.source, args.output, args.lock)


if __name__ == "__main__":
    main()
