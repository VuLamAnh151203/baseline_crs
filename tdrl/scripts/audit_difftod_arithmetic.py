#!/usr/bin/env python3
"""Compare policy arithmetic to pinned source on CPU, never run hosted entrypoints."""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

from tdrl.diffusion import analytic_weights, loglinear_noise, sample_analytic, semantic_constraints
from tdrl.diffusion import score_entropy as local_entropy
from tdrl.provenance import file_hash


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(source, lock):
    if (
        lock["schema_version"] != 1
        or lock["revision"] != "fd46b2887a9c0d9e0e413623e79c72ee32b9b7c1"
    ):
        raise ValueError("unknown DiffTOD source contract")
    for name, expected in lock["files"].items():
        if file_hash(source / name) != expected:
            raise ValueError("DiffTOD source hash mismatch: " + name)
    categorical = load("native_catsample", source / "catsample.py")
    with patch.dict(sys.modules, {"catsample": categorical}):
        graph_module = load("native_graph", source / "graph_lib.py")
    noise = load("native_noise", source / "noise_lib.py").LogLinearNoise()
    graph = graph_module.Absorbing(4)
    clean = torch.tensor([[0, 1, 2], [3, 2, 1]])
    corrupted = torch.tensor([[4, 1, 4], [3, 4, 4]])
    time = torch.tensor([[0.15], [0.75]])
    sigma, rate = noise(time)
    local_sigma, local_rate = loglinear_noise(time)
    torch.testing.assert_close(sigma, local_sigma, rtol=0, atol=0)
    torch.testing.assert_close(rate, local_rate, rtol=0, atol=0)
    logits = torch.linspace(-2, 1, 30).reshape(2, 3, 5).requires_grad_()
    native_loss = graph.score_entropy(logits, sigma, corrupted, clean)
    native_gradient = torch.autograd.grad(native_loss.sum(), logits)[0]
    local_loss = local_entropy(logits, sigma, corrupted, clean, 4)
    local_gradient = torch.autograd.grad(local_loss.sum(), logits)[0]
    torch.testing.assert_close(native_loss, local_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(native_gradient, local_gradient, rtol=1e-6, atol=1e-6)
    delta = torch.tensor([[0.1], [0.4]])
    native_weights = graph.staggered_score(logits.exp(), delta) * graph.transp_transition(
        corrupted, delta
    )
    local_weights = analytic_weights(logits.exp(), corrupted, delta, 4)
    torch.testing.assert_close(native_weights, local_weights, rtol=0, atol=0)

    constraints = semantic_constraints([0, 1], [2, 3], length=200, vocabulary=4)
    calls = []

    class Policy:
        def eval(self):
            return self

        def __call__(self, tokens, sigma):
            calls.append(tokens.clone())
            scores = torch.arange(1, 6, dtype=torch.float32).log().repeat(*tokens.shape, 1)
            # Native SEDD fixes the observed token's log score to zero.
            return scores.scatter(-1, tokens[..., None], 0.0)

    # Only the upstream arithmetic helper is imported. No transformer weights,
    # test resource, network operation or LLM inference is used in this audit.
    mutils = load("native_model_utils", source / "model/utils.py")
    model_module = types.ModuleType("model")
    model_module.utils = mutils
    with patch.dict(sys.modules, {"catsample": categorical, "model": model_module}):
        sampling = load("native_sampling", source / "sampling.py")
    native_sampler = sampling.get_pc_sampler(
        graph,
        noise,
        (2, 200),
        "analytic",
        4,
        device=torch.device("cpu"),
        proj_fun=constraints.project,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        native_tokens = constraints.project(native_sampler(Policy()))
        native_trace = [t.tolist() for t in calls]
        calls.clear()
        torch.manual_seed(17)
        local_tokens = sample_analytic(Policy(), constraints, 4, 2, "cpu")
        local_trace = [t.tolist() for t in calls]
    assert native_trace == local_trace and torch.equal(native_tokens, local_tokens)
    return {
        "schema_version": 1,
        "scope": "CPU pure arithmetic/source fidelity only; not H100 evidence",
        "source": lock,
        "sigma": sigma.tolist(),
        "rate": rate.tolist(),
        "clean": clean.tolist(),
        "corrupted": corrupted.tolist(),
        "logits": logits.tolist(),
        "entropy": native_loss.tolist(),
        "entropy_gradient": native_gradient.tolist(),
        "analytic_delta": delta.tolist(),
        "analytic_weights": native_weights.tolist(),
        "sample_seed": 17,
        "sample_steps": 4,
        "sample_tokens": native_tokens.tolist(),
        "sample_model_calls": len(native_trace),
        "exact_sample_trace_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("configs/difftod-reference.lock.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fixture(args.source, json.loads(args.lock.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "exact_sample_trace_match": True,
                "output": str(args.output),
                "sha256": file_hash(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
