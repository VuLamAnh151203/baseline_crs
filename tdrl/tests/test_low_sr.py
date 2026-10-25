"""Bug regressions and exact CPU geometry tests; not experimental SR evidence."""

import math

import pytest
import torch

from tdrl.contracts import Message
from tdrl.graph import DirectedGraph, Reference
from tdrl.reachability import IQE, MRNFixed, interval_union
from tdrl.selection import choose_head, validate_selection
from tdrl.state import DialogueState


@pytest.mark.parametrize("kind", [MRNFixed, IQE])
def test_head_refactor_matches_original_values_and_gradients(kind):
    torch.manual_seed(17)
    h = kind(input_dim=8)
    a, b = torch.randn(6, 8), torch.randn(6, 8)
    if kind is MRNFixed:
        sym = h.sym(a) - h.sym(b)
        legacy = torch.linalg.vector_norm(sym, dim=-1) / math.sqrt(sym.shape[-1])
        legacy = legacy + (h.asym(a) - h.asym(b)).clamp_min(0).amax(-1)
    else:
        c = interval_union(
            h.encoder(a).unflatten(-1, (h.components, h.component_dim)),
            h.encoder(b).unflatten(-1, (h.components, h.component_dim)),
        )
        alpha = h.mix.sigmoid()
        legacy = alpha * c.amax(-1) + (1 - alpha) * c.mean(-1)
    legacy.sum().backward()
    gradients = {n: p.grad.clone() for n, p in h.named_parameters() if p.grad is not None}
    h.zero_grad(set_to_none=True)
    current = h.distance(a, b)
    assert torch.equal(current, legacy)
    current.sum().backward()
    assert all(torch.equal(p.grad, gradients[n]) for n, p in h.named_parameters())


@pytest.mark.parametrize("kind", [MRNFixed, IQE])
def test_graph_caches_reference_projection_and_paths_without_changing_order(kind, monkeypatch):
    h = kind(input_dim=8)
    states = torch.randn(4, 8)
    refs = [
        Reference(str(i), DialogueState((Message("user", "hello"),)), states[i], "strategy", "d")
        for i in range(4)
    ]
    calls = []
    encode = h.encode
    monkeypatch.setattr(h, "encode", lambda x: (calls.append(x.shape), encode(x))[1])
    g = DirectedGraph(refs, h, 100.0, {}).build(block_size=2)
    assert len(calls) == 1  # All references encoded once, rather than once per block/query.
    found = g.retrieve(states[0], ["3"])
    g.retrieve(states[1], ["3"])
    assert len(calls) == 3  # Only the two live vectors add encoding calls.
    assert g.paths_to_goals(["3"]) is g.paths_to_goals(["3"])
    with torch.no_grad():
        expected = h.distance(states[0], states).tolist()
    assert dict((n, cost) for cost, n, _ in found) == pytest.approx(
        dict(zip(map(str, range(4)), expected, strict=True))
    )
    paths = g.paths_to_goals(["3"])
    g.add_edge("0", "3", 0.0)
    assert g.paths_to_goals(["3"]) is not paths


def zero_candidate():
    return {
        "development_sr": {"TopDial": 0.0, "DuRecDial2.0": 0.0},
        "development_successes": {"TopDial": 0, "DuRecDial2.0": 0},
        "h100_seconds_per_success": None,
        "parameters": 100,
        "invariants_passed": True,
        "stability_passed": True,
        "provenance_passed": True,
    }


def test_zero_success_is_undefined_cost_not_an_automatic_pipeline_crash():
    from test_core import candidate

    zero = zero_candidate()
    positive = candidate(0.0625)
    assert choose_head({"MRNFixed": zero, "IQE": positive}, [-0.1, 0.1]) == "IQE"
    assert choose_head({"MRNFixed": positive, "IQE": zero}, [-0.1, 0.1]) == "MRNFixed"
    with pytest.raises(ValueError, match="all valid heads"):
        choose_head({"MRNFixed": zero, "IQE": zero}, [0.0, 0.0])
    with pytest.raises(ValueError, match="verified zero"):
        choose_head(
            {"MRNFixed": {**zero, "development_successes": None}, "IQE": positive}, [-0.1, 0.1]
        )


def test_recovery_lock_schema_rejects_unknown_versions_and_verifies_decision(tmp_path):
    import json

    from test_core import candidate

    from tdrl.provenance import digest
    from tdrl.selection import verify_selection, write_selection

    payload = {
        "schema_version": 3,
        "scope": {
            "methods": ["PPDPP", "TDRL-R", "TDRL"],
            "claim": "three_method_automated_comparison",
        },
        "selected_head": "IQE",
        "candidates": {
            "MRNFixed": zero_candidate(),
            "IQE": {
                **candidate(0.0625),
                "development_successes": {"TopDial": 1, "DuRecDial2.0": 1},
            },
        },
        "difference_ci": [-0.1, 0.1],
        "thresholds": {d: {"H_edge": 1.0, "H_cover": 2.0} for d in ["TopDial", "DuRecDial2.0"]},
        "prompts": {"p": "h"},
        "models": {"m": "h"},
        "budgets": {"b": 1},
        "baseline_adaptations": {"PPDPP": {"status": "validated", "artifact_sha256": "h"}},
        "scenario_manifests": {"s": "h"},
        "config_sha256": "h",
        "package_lock_sha256": "h",
        "code": {"commit": "h"},
        "gate_evidence": {str(i): "h" for i in range(4)},
        "recovery_amendment": {"path": "amendment.json", "sha256": "h", "source_job_id": "57711"},
    }
    path = tmp_path / "selection.lock"
    sha = write_selection(path, payload)
    assert verify_selection(path, sha) == payload
    assert json.loads(path.read_text())["sha256"] == digest(payload)
    with pytest.raises(ValueError):
        validate_selection({**payload, "schema_version": 99})


def test_graph_guidance_is_strategy_only_not_foreign_entity_facts():
    from tdrl.training import make_graph

    records = [
        {
            "split": "train",
            "guidance": "Alice recommends Foreign Movie and its 9.9 rating.",
            "target": ["Movie recommendation", "Foreign Movie"],
            "action": "recommend",
            "state": DialogueState((Message("user", "I like films."),)).serialize(),
            "node_id": "n",
            "dialogue_id": "d",
            "goal_reference": True,
        }
    ]
    graph, goals = make_graph(
        records, torch.zeros(1, 4), MRNFixed(input_dim=4), 1.0, 8, 17, "Movie recommendation"
    )
    assert goals == ["n"]
    guidance = graph.references["n"].guidance
    assert "recommend" in guidance
    assert "Alice" not in guidance and "Foreign Movie" not in guidance and "9.9" not in guidance


def test_clear_refusals_stop_without_changing_acceptance_rule():
    from tdrl.acceptance import ACCEPTANCE_VERSION, classify_user
    from tdrl.autonomy import explicit_refusal

    for text in [
        "I'm not really interested in that movie.",
        "I don't need any other recommendations.",
    ]:
        state = DialogueState((Message("user", text),))
        assert explicit_refusal(state)
        assert classify_user(state, ["movie", "The Star"]).label == "rejection"
    assert ACCEPTANCE_VERSION == 1
    assert not classify_user(
        DialogueState((Message("user", "The Star sounds nice."),)), ["movie", "The Star"]
    ).accepted


def test_shared_prompts_preserve_preferences_but_do_not_treat_acceptances_as_whitelist():
    from tdrl.evaluation import USER_POLICY
    from tdrl.planner import ASSISTANT_POLICY

    assert "not an exhaustive whitelist" in USER_POLICY
    assert "Keep stated facts" in USER_POLICY
    assert "do not copy reference people" in ASSISTANT_POLICY
