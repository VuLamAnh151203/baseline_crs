import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from tdrl.audit import static_audit
from tdrl.backend import LocalBackend, loopback_url, require_h100_slurm
from tdrl.budget import experiment_matrix, feasibility
from tdrl.contracts import Decoding, Embeddings, Generation, Message, ModelProvenance
from tdrl.data import duplicate_audit, prepare_canonical
from tdrl.graph import DirectedGraph, Reference
from tdrl.learning import CostConfig, ReachabilityLearner, bellman_targets
from tdrl.metrics import paired_bootstrap, validate_records
from tdrl.planner import ConditionalPlanner, PlannerConfig, derived_seed
from tdrl.provenance import COMPUTE_FIELDS, PROVENANCE_FIELDS, Journal, digest, validate_manifest
from tdrl.reachability import IQE, MRNFixed, assert_invariants, interval_union
from tdrl.selection import choose_head, verify_selection, write_selection
from tdrl.state import DialogueState, observed_prefix

ROOT = Path(__file__).resolve().parents[1]
PIN = ModelProvenance("Qwen/Qwen3.5-9B", "a" * 40, {"weights": "b" * 64}, "c" * 64)


@pytest.mark.parametrize("head_class", [MRNFixed, IQE])
@pytest.mark.parametrize(
    "dtype,atol,rtol", [(torch.float32, 1e-5, 1e-5), (torch.bfloat16, 2e-3, 2e-2)]
)
def test_directed_invariants_and_finite_training(head_class, dtype, atol, rtol):
    torch.manual_seed(29)
    head = head_class(input_dim=32).to(dtype)
    x = torch.randn(16, 32).to(dtype)
    assert_invariants(head, x, atol, rtol)
    distance = head.distance(x[:8], x[8:]).float().sum()
    distance.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


def test_interval_union_overlap_ties_and_direction():
    x = torch.tensor([[0.0, 1.0, 6.0], [1.0, 1.0, 1.0]])
    y = torch.tensor([[3.0, 4.0, 7.0], [1.0, 1.0, 1.0]])
    assert interval_union(x, y).tolist() == [5.0, 0.0]
    assert interval_union(y, x).tolist() == [0.0, 0.0]


def test_symmetric_and_unconstrained_impostors_are_rejected():
    class Symmetric(torch.nn.Module):
        def distance(self, source, target):
            return (source - target).norm(dim=-1)

    class Squared(torch.nn.Module):
        def distance(self, source, target):
            return (source - target).square().sum(-1)

    states = torch.tensor([[0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="symmetric"):
        assert_invariants(Symmetric(), states, atol=2e-3, rtol=2e-2)
    with pytest.raises(ValueError, match="triangle"):
        assert_invariants(Squared(), states, atol=2e-3, rtol=2e-2)


def test_sdk_exception_only_for_mandatory_runtime_dependencies():
    from tdrl.audit import dependency_audit

    assert dependency_audit(ROOT / "runtime/uv.lock", allow_local_vllm=True)["status"] == "passed"
    assert dependency_audit(ROOT / "runtime/uv.lock")["status"] == "blocked"
    with pytest.raises(ValueError, match="pinned"):
        dependency_audit(ROOT / "uv.lock", allow_local_vllm=True)


def test_bellman_goal_predecessor_multistep_detachment_and_failed_terminal():
    head = MRNFixed(input_dim=8)
    nxt, goals = torch.randn(3, 8), torch.randn(3, 8)
    yes = torch.ones(3, dtype=torch.bool)
    y = bellman_targets(
        head,
        nxt,
        goals,
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
        yes,
        CostConfig(),
    )
    assert y[0] == 0 and y[1] == 1
    assert torch.allclose(y[2], 1 + head.distance(nxt[2], goals[2]))
    assert not y.requires_grad
    with pytest.raises(ValueError, match="separation"):
        bellman_targets(head, nxt, goals, ~yes, ~yes, ~yes, CostConfig())
    learner = ReachabilityLearner(head)
    loss = learner.step(torch.randn(3, 8), nxt, goals, ~yes, ~yes, ~yes)
    assert loss >= 0 and learner.steps == 1
    assert all(p.grad is None for p in learner.target.parameters())


def test_resume_target_optimizer_rng():
    torch.manual_seed(17)
    learner = ReachabilityLearner(MRNFixed(input_dim=8))
    data = [torch.randn(4, 8) for _ in range(3)]
    no, yes = torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)
    learner.step(*data, no, no, yes)
    import copy

    checkpoint = copy.deepcopy(learner.state_dict())
    expected = learner.step(*data, no, no, yes)
    restored = ReachabilityLearner(MRNFixed(input_dim=8))
    restored.load_state_dict(checkpoint)
    assert expected == restored.step(*data, no, no, yes)
    for a, b in zip(learner.head.parameters(), restored.head.parameters(), strict=True):
        assert torch.equal(a, b)


def test_serialization_has_no_target_or_future():
    messages = [Message("user", "hello"), Message("assistant", "future_secret")]
    s = observed_prefix(messages, 1).serialize()
    assert "future_secret" not in s and "target" not in s
    assert s == DialogueState((messages[0],)).serialize()
    with pytest.raises(ValueError):
        DialogueState((Message("system", "goal_secret"),))


class OrderedHead(torch.nn.Module):
    head_type = "unit_fake"
    checkpoint_hash = "fake"
    encoder_provenance = {}

    def distance(self, source, target):
        return (target[..., 0] - source[..., 0]).clamp_min(0)


def graph_fixture():
    state = DialogueState((Message("user", "hello"),))
    refs = [
        Reference(str(i), state, torch.tensor([float(i)]), "ask a preference", "train1")
        for i in range(3)
    ]
    return DirectedGraph(refs, OrderedHead(), 0.5, {})


def test_directed_graph_edges_no_reverse_and_retrieval_order():
    graph = graph_fixture()
    graph.add_edge("0", "1", 1)
    graph.add_edge("1", "2", 1)
    assert graph.shortest_path("0", ["2"]) == (2, ["0", "1", "2"])
    assert graph.shortest_path("2", ["0"])[1] == []
    scores = graph.retrieve(torch.tensor([-1.0]), ["2"])
    assert scores[0][0] == 1 and scores[0][1] == "0"
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            graph.add_edge("0", "1", bad)
    with pytest.raises(ValueError, match="training"):
        DirectedGraph(
            [Reference("x", graph.references["0"].state, torch.zeros(1), "g", "d", "dev")],
            OrderedHead(),
            1,
            {},
        )


def test_head_mutation_invalidates_graph():
    head = MRNFixed(input_dim=4)
    state = DialogueState(())
    g = DirectedGraph([Reference("a", state, torch.ones(4), "g", "train1")], head, 1, {})
    with torch.no_grad():
        next(head.parameters()).add_(1)
    with pytest.raises(ValueError, match="changed"):
        g.retrieve(torch.ones(4), ["a"])


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/v1",
        "http://example.com/v1",
        "http://localhost/v1",
        "http://127.0.0.1.example.com/v1",
        "http://user:pass@127.0.0.1/v1",
        "http://0.0.0.0/v1",
        "http://169.254.169.254/v1",
        "http://127.0.0.1/v1?host=remote",
    ],
)
def test_nonloopback_rejected(url):
    with pytest.raises(ValueError):
        loopback_url(url)


def test_loopback_and_slurm_guard(monkeypatch):
    assert loopback_url("http://127.0.0.1:8000/v1")
    assert loopback_url("http://[::1]:8000/v1")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        require_h100_slurm()
    assert not static_audit(ROOT)


def test_embedding_order_shape_and_normalization(monkeypatch):
    model = ModelProvenance("Qwen/Qwen3-Embedding-8B", "a" * 40, {"w": "b" * 64}, "c" * 64)
    backend = LocalBackend("http://127.0.0.1:8001/v1", model, "embedding")
    a, b = [2.0] + [0.0] * 4095, [0.0, 3.0] + [0.0] * 4094
    response = {
        "model": model.model_id,
        "data": [{"index": 1, "embedding": b}, {"index": 0, "embedding": a}],
    }
    monkeypatch.setattr(backend, "_post", lambda *args: (response, 0.1))
    output = backend.embed(["a", "b"], "task", "batch", "request")
    assert output.vectors.shape == (2, 4096)
    assert output.vectors[0, 0] == 1 and output.vectors[1, 1] == 1
    response["data"][1]["index"] = 1
    with pytest.raises(ValueError, match="order"):
        backend.embed(["a", "b"], "task", "batch", "request")


def test_generation_nonthinking_final_only_and_usage(monkeypatch):
    backend = LocalBackend("http://127.0.0.1:8000/v1", PIN, "generation")
    response = {
        "model": PIN.model_id,
        "choices": [{"message": {"content": "Hello!"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }

    def post(endpoint, payload):
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        return response, 0.1

    monkeypatch.setattr(backend, "_post", post)
    out = backend.generate([Message("user", "hello")], "assistant", Decoding(), 17, "r")
    assert out.text == "Hello!" and out.total_tokens == 3
    response["choices"][0]["message"]["content"] = "<think>secret</think>Hello"
    with pytest.raises(ValueError, match="final text"):
        backend.generate([], "assistant", Decoding(), 17, "r2")


def test_journal_resume_and_mismatch(tmp_path):
    j = Journal(tmp_path, {"model_revision": "a", "serialization": "v1"})
    j.write("r1", {"seed": 17}, {"text": "hello"})
    resumed = Journal(tmp_path, {"model_revision": "a", "serialization": "v1"})
    assert resumed.read("r1", {"seed": 17}) == {"text": "hello"}
    with pytest.raises(ValueError):
        resumed.read("r1", {"seed": 29})
    with pytest.raises(ValueError):
        Journal(tmp_path, {"model_revision": "b", "serialization": "v1"})


class FakeGeneration:
    def __init__(self):
        self.requests = []

    def generate(self, messages, role, decoding, seed, request_id):
        r = Generation(
            "I like comedies." if role == "rollout_user" else "Which genres do you like?",
            2,
            3,
            5,
            seed,
            decoding,
            0.1,
            0,
            PIN,
            digest([asdict(m) for m in messages]),
            request_id,
            role,
        )
        self.requests.append(r)
        return r


class FakeEmbedding:
    def __init__(self):
        self.requests = []

    def embed(self, texts, instruction, batch_id, request_id):
        r = Embeddings(
            torch.tensor([[-1.0]]),
            PIN,
            "instruction",
            "prep",
            len(texts),
            0.1,
            3,
            None,
            batch_id,
            request_id,
        )
        self.requests.append(r)
        return r


def test_planner_separate_thresholds_recovery_counters_seeds_and_refusal():
    graph = graph_fixture()
    graph.add_edge("0", "2", 2)
    generation, embedding = FakeGeneration(), FakeEmbedding()
    planner = ConditionalPlanner(
        generation,
        embedding,
        graph,
        PlannerConfig(h_cover=0.1, candidates=2, lookahead=2),
        simulator_prompt="frozen-user",
    )
    plan = planner.act(graph.references["0"].state, ["2"], 17, "dialogue/turn/0")
    assert not plan.covered and plan.selected_candidate == 0
    assert plan.compute["generation_requests"] == 8
    assert plan.compute["embedding_requests"] == 3
    assert plan.compute["total_tokens"] == 40
    assert graph.h_edge == 0.5
    assert len({r.seed for r in generation.requests}) == 8
    assert derived_seed(17, "d", "candidate", 0) != derived_seed(17, "d", "candidate", 1)
    refused = DialogueState((Message("user", "No thanks, I'm not interested."),))
    plan = planner.act(refused, ["2"], 17, "refused")
    assert plan.compute["refusal_guard"] and plan.compute["generation_requests"] == 0


def test_data_test_lock_and_membership(tmp_path):
    path = tmp_path / "input.jsonl"
    row = {
        "id": "d1",
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "future"}],
        "profile": {},
        "knowledge": [],
        "target": "secret",
    }
    path.write_text(json.dumps(row) + "\n")
    provenance = {
        k: "hash"
        for k in ("source_url", "source_revision", "raw_sha256", "license", "adapter_sha256")
    }
    provenance.update(
        schema_version=1,
        split_membership_sha256=digest(["d1"]),
        official_split="train",
        transformations=[],
    )
    out = tmp_path / "prepared"
    prepare_canonical(path, out, "TopDial", "train", provenance)
    state = json.loads((out / "states.jsonl").read_text())["state"]
    assert "future" not in state and "secret" not in state
    with pytest.raises(ValueError, match="locked"):
        prepare_canonical(path, tmp_path / "test", "TopDial", "test", provenance)
    assert duplicate_audit({"train": [row], "dev": [row]})[0]["exact"]


def candidate(sr, cost=10, parameters=10):
    return {
        "invariants_passed": True,
        "stability_passed": True,
        "provenance_passed": True,
        "development_sr": {"TopDial": sr, "DuRecDial2.0": sr},
        "h100_seconds_per_success": cost,
        "parameters": parameters,
    }


def test_selection_rule_and_immutable_lock(tmp_path):
    candidates = {"MRNFixed": candidate(0.5), "IQE": candidate(0.6)}
    assert choose_head(candidates, (0.02, 0.2)) == "IQE"
    assert choose_head(candidates, (-0.01, 0.2)) == "MRNFixed"
    candidates["IQE"]["h100_seconds_per_success"] = 9
    assert choose_head(candidates, (-0.01, 0.2)) == "IQE"
    payload = {
        "schema_version": 1,
        "selected_head": "IQE",
        "candidates": candidates,
        "difference_ci": [-0.01, 0.2],
        "thresholds": {d: {"H_edge": 1, "H_cover": 2} for d in ("TopDial", "DuRecDial2.0")},
        "prompts": {"p": "h"},
        "models": {"m": "h"},
        "budgets": {"b": 1},
        "baseline_adaptations": {
            name: {"status": "validated", "artifact_sha256": "h"}
            for name in ("PPDPP", "DPDP", "TRIP", "DiffTOD")
        },
        "scenario_manifests": {"s": "h"},
        "config_sha256": "h",
        "package_lock_sha256": "h",
        "code": {"commit": "h"},
        "gate_evidence": {str(i): "h" for i in range(4)},
    }
    p = tmp_path / "selection.lock"
    sha = write_selection(p, payload)
    assert verify_selection(p, sha) == payload
    with pytest.raises(FileExistsError):
        write_selection(p, payload)
    with pytest.raises(ValueError):
        verify_selection(p, "wrong")


def test_manifest_versions_partial_and_finite():
    m = {
        "schema_version": 1,
        "status": "partial",
        "run_id": "r",
        "stage": "smoke",
        "started_at": "now",
        "ended_at": None,
        "provenance": dict.fromkeys(PROVENANCE_FIELDS),
        "compute": dict.fromkeys(COMPUTE_FIELDS),
        "outputs": {},
    }
    validate_manifest(m)
    m["status"] = "complete"
    with pytest.raises(ValueError):
        validate_manifest(m)
    m["status"] = "partial"
    m["compute"]["h100_seconds"] = float("nan")
    with pytest.raises(ValueError):
        validate_manifest(m)
    m["schema_version"] = 99
    with pytest.raises(ValueError, match="version"):
        validate_manifest(m)


def test_budget_matrix_and_missing_throughput():
    cells = experiment_matrix({"TopDial": ["test_seen", "test_unseen"], "DuRecDial2.0": ["test"]})
    assert len(cells) == 324
    assert {c["method"] for c in cells} >= {"PPDPP", "DPDP", "TRIP", "DiffTOD"}
    cell = cells[0]
    measured = {f"{cell['method']}/{cell['budget']}/{cell['protocol']}/{cell['dataset']}": 60}
    assert feasibility([cell], {"TopDial/test_seen": 100}, measured, 24)["fits"] is False
    with pytest.raises(KeyError):
        feasibility(cells, {}, {}, 480)


def test_bootstrap_preserves_scenario_seed_clusters_and_completion():
    a = [[1, 1, 1], [0, 0, 0], [1, 0, 1]]
    result = paired_bootstrap(a, a)
    assert result["difference"] == 0 and result["ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="partial"):
        validate_records([], ["a", "b"], [17, 29, 43], "lock")


def test_inference_transport_uses_fresh_tcp_connections_without_automatic_retries(
    monkeypatch, tmp_path
):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx

    peers = []
    calls = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            peers.append(self.client_address)
            calls.append(self.path)
            if len(calls) == 3:
                self.close_connection = True
                return
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr("tdrl.backend.require_h100_slurm", lambda: {})
    ledger = tmp_path / "ledger"
    backend = LocalBackend(
        f"http://127.0.0.1:{server.server_port}/v1", PIN, "generation", ledger=ledger
    )
    try:
        for _ in range(2):
            assert backend._post("/probe", {})[0] == {"ok": True}
        assert peers[0] != peers[1]
        pending = backend._begin("failed-request", {"input": "fixture"})
        with pytest.raises(httpx.RemoteProtocolError):
            backend._post("/probe", {})
        assert len(calls) == 3
        assert pending.exists()
        with pytest.raises(ValueError, match="unresolved"):
            backend._begin("failed-request", {"input": "fixture"})
    finally:
        backend.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
