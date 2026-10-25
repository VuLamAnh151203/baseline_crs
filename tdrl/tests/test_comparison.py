"""CPU fixtures verify control flow/mechanisms, never establish H100 evidence."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_core import FakeEmbedding, FakeGeneration, candidate
from torch import nn

from tdrl.acceptance import classify_user
from tdrl.comparison import Comparison, validate_config
from tdrl.contracts import Message
from tdrl.evaluation import run_dialogue
from tdrl.near_duplicates import audit_train_dev, shingles
from tdrl.official_data import ACTION_TAXONOMY, adapt, read_release
from tdrl.planner import ConditionalPlanner, PlannerConfig
from tdrl.ppdpp import PPDPPPolicy, reinforce_loss
from tdrl.provenance import digest
from tdrl.selection import validate_selection
from tdrl.state import DialogueState
from tdrl.training import PairSampler, state_records

ROOT = Path(__file__).resolve().parents[1]


def scenario():
    return {
        "id": "fixture",
        "profile": {},
        "personality": {},
        "knowledge": [],
        "target": ["movie", "The Star"],
        "messages": [{"role": "user", "content": "I enjoy comedy."}],
        "assistant_actions": [None],
    }


@pytest.mark.parametrize(
    "text,accepted",
    [
        ("I will watch The Star tonight.", True),
        ("I've already seen The Star.", False),
        ("Maybe I will watch The Star.", False),
        ("I will not watch The Star.", False),
        ("Thanks for mentioning The Star.", False),
        ("I will watch The Starship tonight.", False),
        ("The Star sounds nice. I will watch a comedy.", False),
    ],
)
def test_explicit_prospective_target_acceptance(text, accepted):
    state = DialogueState((Message("user", text),))
    assert classify_user(state, ["movie", "The Star"]).accepted is accepted


def test_refusal_cannot_be_overwritten_by_later_acceptance():
    state = DialogueState(
        (
            Message("user", "I do not want recommendations."),
            Message("assistant", "Understood."),
            Message("user", "I will watch The Star."),
        )
    )
    assert not classify_user(state, ["movie", "The Star"]).accepted


def test_du_recdial_excludes_future_aligned_knowledge():
    row = {
        "goal": "[1]Q&A(Topic)-->[2]Movie recommendation(The Star)-->[3]Say goodbye",
        "conversation": ["[1]Tell me about films", "[1]Here are films", "[2]future"],
        "knowledge": [["initial"], ["future-answer"], ["future-target"]],
        "user_profile": {},
    }
    record = adapt("DuRecDial2.0", row, 0)
    assert record["messages"][0] == {"role": "user", "content": "Tell me about films"}
    assert record["target"] == ["Movie recommendation", "The Star"]
    assert record["knowledge"] == [["initial"]]
    row["goal"] = "[1]Greetings-->[2]Movie recommendation(The Star)-->[3]Say goodbye"
    assert adapt("DuRecDial2.0", row, 0)["messages"][0]["role"] == "assistant"
    assert adapt("DuRecDial2.0", row, 0)["knowledge"] == [["initial"], ["future-answer"]]


def test_test_loader_fails_before_touching_files():
    with pytest.raises(ValueError, match="sealed"):
        read_release(Path("/nonexistent"), "TopDial", "test_seen")


def test_state_record_never_contains_future_guidance():
    s = scenario()
    s["messages"].append({"role": "assistant", "content": "UNIQUE FUTURE The Star"})
    s["assistant_actions"].append("recommend")
    records = state_records([s], "TopDial", "train")
    assert "UNIQUE FUTURE" not in records[0]["state"]
    assert records[-1]["failed_terminal"]


def test_prefix_near_duplicate_filter_matches_exhaustive_jaccard():
    rng = random.Random(17)

    def record(i, words):
        return {"id": str(i), "messages": [{"content": " ".join(words)}]}

    words = [f"word{i}" for i in range(150)]
    train = [record(i, rng.sample(words, rng.randrange(30, 120))) for i in range(25)]
    dev = [
        record("d" + str(i), r["messages"][0]["content"].split()) for i, r in enumerate(train[:6])
    ]
    dev += [record("other", rng.sample(words, 50))]
    for threshold in (0.5, 0.8, 0.9, 1.0):
        expected = set()
        for left in train:
            for right in dev:
                a, b = shingles(left["messages"]), shingles(right["messages"])
                if len(a & b) / len(a | b) >= threshold:
                    expected.add((left["id"], right["id"]))
        found = {
            (r["train_id"], r["dev_id"])
            for r in audit_train_dev(train, dev, threshold)["collisions"]
        }
        assert found == expected


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 4)

    def forward(self, input_ids, attention_mask):
        pooled = self.embedding(input_ids).mean(1)
        return (None, pooled)


def test_ppdpp_keeps_trainable_policy_ce_sampling_and_reinforce():
    torch.manual_seed(17)
    model = PPDPPPolicy(FakeBackbone(), 4)
    assert model.actions == sorted(ACTION_TAXONOMY)
    assert model.dropout.p == 0.5
    ids = torch.tensor([[1, 2], [3, 4]])
    loss = model.supervised_loss(ids, torch.ones_like(ids), torch.tensor([0, 5]))
    loss.backward()
    assert model.policy.embedding.weight.grad.abs().sum() > 0
    model.eval()
    chosen, logs = model.select(ids, torch.ones_like(ids), evaluate=True)
    assert torch.equal(chosen, model.logits(ids, torch.ones_like(ids)).argmax(-1))
    model.zero_grad(set_to_none=True)
    _, logs = model.select(ids, torch.ones_like(ids), torch.Generator().manual_seed(29))
    reinforce_loss(list(logs), [-0.05, 1.0], 0.99).backward()
    assert model.classifier.weight.grad.abs().sum() > 0


def test_reinforce_discount_and_detached_native_normalization():
    logs = torch.tensor([-0.3, -0.7, -0.4], requires_grad=True)
    rewards = [-0.05, -0.05, 1.0]
    returns = torch.tensor([-0.05 + 0.99 * (-0.05 + 0.99), -0.05 + 0.99, 1.0])
    normalized = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float32).eps)
    loss = reinforce_loss(list(logs), rewards, 0.99)
    assert torch.allclose(loss, -(logs * normalized).sum())
    loss.backward()
    assert torch.allclose(logs.grad, -normalized)


def test_failed_terminal_is_never_a_reachable_goal():
    records = [
        {"dataset": "d", "dialogue_id": str(d), "failed_terminal": t == 2}
        for d in range(3)
        for t in range(3)
    ]
    sampler = PairSampler(records, 17)
    source, successor, goals, self_goal, next_goal, reachable = sampler.sample(2000)
    for s, n, g, sg, ng, positive in zip(
        source, successor, goals, self_goal, next_goal, reachable, strict=True
    ):
        assert not records[g]["failed_terminal"]
        assert sg == (s == g) and ng == (n == g)
        if positive:
            assert not records[s]["failed_terminal"]
        elif records[s]["failed_terminal"]:
            assert s == n


def test_dialogue_resume_pairing_and_one_call_matched_budget(tmp_path):
    generation, embedding = FakeGeneration(), FakeEmbedding()
    planner = ConditionalPlanner(
        generation, embedding, None, PlannerConfig(0.0, generation_call_cap=1)
    )
    contract = {"phase": "test-control", "method": "TDRL", "selection_sha256": "a" * 64}
    s = scenario()
    partial = run_dialogue(
        s,
        planner,
        generation,
        embedding,
        "profile",
        17,
        tmp_path,
        contract,
        turn_budget=3,
        interrupt_after=1,
    )
    assert partial["status"] == "partial"
    first = partial["turns_trace"][0]
    complete = run_dialogue(
        s, planner, generation, embedding, "profile", 17, tmp_path, contract, turn_budget=3
    )
    assert complete["status"] == "complete" and complete["turns"] == 3
    assert complete["turns_trace"][0] == first
    assert complete["generation_requests"] == 6
    assert complete["embedding_requests"] == 3
    assert complete["total_tokens"] == 30
    assert all(t["plan"]["compute"]["recovery_budget_exhausted"] for t in complete["turns_trace"])
    count = len(generation.requests)
    assert (
        run_dialogue(
            s, planner, generation, embedding, "profile", 17, tmp_path, contract, turn_budget=3
        )
        == complete
    )
    assert len(generation.requests) == count
    with pytest.raises(ValueError, match="contract"):
        run_dialogue(
            s,
            planner,
            generation,
            embedding,
            "profile",
            17,
            tmp_path,
            {**contract, "method": "PPDPP"},
            turn_budget=3,
        )
    # Method names/paths differ, but the final-response and real-user seeds are paired.
    other_gen, other_emb = FakeGeneration(), FakeEmbedding()
    other = ConditionalPlanner(
        other_gen, other_emb, None, PlannerConfig(0.0, routing="retrieval_only")
    )
    result = run_dialogue(
        s,
        other,
        other_gen,
        other_emb,
        "profile",
        17,
        tmp_path / "other",
        {**contract, "method": "TDRL-R"},
        turn_budget=3,
    )
    assert [[r["seed"] for r in t["generation"]] for t in result["turns_trace"]] == [
        [r["seed"] for r in t["generation"]] for t in complete["turns_trace"]
    ]


def test_scoped_selection_requires_ppdpp_evidence_and_fail_closed_versions():
    payload = {
        "schema_version": 1,
        "selected_head": "MRNFixed",
        "candidates": {"MRNFixed": candidate(0.5), "IQE": candidate(0.5)},
        "difference_ci": [0.0, 0.0],
        "thresholds": {d: {"H_edge": 1.0, "H_cover": 2.0} for d in ("TopDial", "DuRecDial2.0")},
        "prompts": {"p": "h"},
        "models": {"m": "h"},
        "budgets": {"b": 1},
        "baseline_adaptations": {"PPDPP": {"status": "validated", "artifact_sha256": "h"}},
        "scenario_manifests": {"s": "h"},
        "config_sha256": "h",
        "package_lock_sha256": "h",
        "code": {"commit": "h"},
        "gate_evidence": {str(i): "h" for i in range(4)},
    }
    payload["schema_version"] = 2
    payload["scope"] = {
        "methods": ["PPDPP", "TDRL-R", "TDRL"],
        "claim": "three_method_automated_comparison",
    }
    payload["baseline_adaptations"] = {"PPDPP": payload["baseline_adaptations"]["PPDPP"]}
    validate_selection(payload)
    payload["schema_version"] = 99
    with pytest.raises(ValueError):
        validate_selection(payload)


def test_comparison_retains_failure_and_never_runs_training_after_bad_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    cfg = ROOT / "configs/comparison.json"
    validate_config(json.loads(cfg.read_text()))
    instance = Comparison(ROOT, tmp_path, cfg)
    called = []

    def stage(name, fn, extra=None):
        called.append(name)
        return SimpleNamespace()

    monkeypatch.setattr(instance, "stage", stage)

    def failed_smoke(*args, **kwargs):
        raise RuntimeError("dual residency failed")

    monkeypatch.setattr("tdrl.comparison.smoke", failed_smoke)
    with pytest.raises(RuntimeError, match="dual residency"):
        instance.run()
    assert called == ["gate0", "gate1"]
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "partial"
    assert digest(instance.cfg)


def test_on_policy_resume_reconstructs_sampled_dropout_gradients(monkeypatch):
    from tdrl.policy_training import replay_policy_actions

    s = scenario()

    def inputs(tokenizer, state, context):
        ids = torch.tensor([[len(state.history) % 10, 1]])
        return ids, torch.ones_like(ids)

    monkeypatch.setattr("tdrl.policy_training.policy_input", inputs)
    torch.manual_seed(29)
    policy = PPDPPPolicy(FakeBackbone(), 4).train()
    before = torch.get_rng_state()
    original = SimpleNamespace(
        policy=policy,
        tokenizer=None,
        context="",
        log_probabilities=[],
        sample_generator=torch.Generator().manual_seed(43),
    )
    history = [Message("user", "I enjoy comedy.")]
    traces = []
    for _ in range(3):
        ids, mask = inputs(None, DialogueState(tuple(history)), "")
        action, log = policy.select(ids, mask, original.sample_generator)
        original.log_probabilities.append(log.squeeze())
        traces.append({"plan": {"compute": {"strategy": policy.actions[int(action.item())]}}})
        history += [Message("assistant", "A suggestion."), Message("user", "Please explain.")]
    expected_values = torch.stack(original.log_probabilities).detach()
    reinforce_loss(original.log_probabilities, [-0.05, -0.05, 1.0]).backward()
    expected_gradients = {name: p.grad.clone() for name, p in policy.named_parameters()}
    policy.zero_grad(set_to_none=True)
    torch.set_rng_state(before)
    resumed = SimpleNamespace(
        policy=policy,
        tokenizer=None,
        context="",
        log_probabilities=[],
        sample_generator=torch.Generator().manual_seed(43),
    )
    stored = {
        "history": [{"role": m.role, "content": m.content} for m in history],
        "turns_trace": traces,
    }
    replay_policy_actions(resumed, stored, s)
    assert torch.equal(torch.stack(resumed.log_probabilities).detach(), expected_values)
    reinforce_loss(resumed.log_probabilities, [-0.05, -0.05, 1.0]).backward()
    for name, p in policy.named_parameters():
        assert torch.equal(p.grad, expected_gradients[name])
    stored["turns_trace"][0]["plan"]["compute"]["strategy"] = "nonexistent"
    torch.set_rng_state(before)
    resumed.sample_generator.manual_seed(43)
    with pytest.raises(ValueError, match="replay mismatch"):
        replay_policy_actions(resumed, stored, s)


def test_generation_actual_ledger_survives_resume_but_cache_is_not_a_call(tmp_path, monkeypatch):
    from test_core import PIN

    from tdrl.backend import LocalBackend
    from tdrl.contracts import Decoding
    from tdrl.provenance import Journal
    from tdrl.stages import counters

    journal = Journal(tmp_path / "journal", {"model": "fixture"})
    ledger = tmp_path / "ledger"
    backend = LocalBackend("http://127.0.0.1:8000/v1", PIN, "generation", journal, ledger)
    response = {
        "model": PIN.model_id,
        "choices": [{"message": {"content": "Hello!"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    monkeypatch.setattr(backend, "_post", lambda *args: (response, 0.1))
    original = backend.generate([Message("user", "hello")], "assistant", Decoding(), 17, "request")
    resumed = LocalBackend("http://127.0.0.1:8000/v1", PIN, "generation", journal, ledger)
    monkeypatch.setattr(resumed, "_post", lambda *args: pytest.fail("replay made an HTTP call"))
    assert (
        resumed.generate([Message("user", "hello")], "assistant", Decoding(), 17, "request")
        == original
    )
    assert not resumed.requests and len(resumed.consumed_requests) == 1
    assert counters(resumed, SimpleNamespace(requests=[]))["generation_requests"] == 1
    resumed._begin("uncertain-request", {"input": "fixture"})
    with pytest.raises(ValueError, match="unresolved"):
        counters(resumed, SimpleNamespace(requests=[]))
    with pytest.raises(ValueError, match="in-flight"):
        resumed._begin("uncertain-request", {"input": "fixture"})


def test_heldout_temporal_labels_and_directed_order_are_independent(tmp_path):
    from tdrl.training import heldout_temporal_diagnostics

    class FixedHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def distance(self, source, target):
            delta = (target - source).squeeze(-1)
            return delta.abs() + (-delta).relu()

    rows = [
        {"dialogue_id": str(d), "node_id": f"{d}:{t}", "split": "dev", "failed_terminal": False}
        for d in range(2)
        for t in range(4)
    ]
    vectors = torch.tensor([[float(t)] for _ in range(2) for t in range(4)])
    result = heldout_temporal_diagnostics(FixedHead(), rows, vectors, tmp_path, pairs=64)
    assert result["label_distance_access"] is False
    assert result["forward_less_than_reverse"] == 1.0
    assert result["remaining_step_pearson"] == pytest.approx(1.0)
    assert all(
        r["forward"] == r["remaining_transitions"] and r["reverse"] == 2 * r["forward"]
        for r in result["rows"]
    )
    rows[0]["split"] = "train"
    with pytest.raises(ValueError, match="development"):
        heldout_temporal_diagnostics(FixedHead(), rows, vectors, tmp_path, pairs=64)


def test_batch_module_entrypoint_executes_static_gate():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "tdrl", "--root", str(ROOT), "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "passed", "errors": []}


def test_runtime_lock_pins_published_cuda12_wheel_and_matching_torch():
    import tomllib

    from tdrl.audit import dependency_audit

    pin = json.loads((ROOT / "configs/runtime-build.lock.json").read_text())
    lock = tomllib.loads((ROOT / "runtime/uv.lock").read_text())
    packages = {
        p["name"]: p
        for p in lock["package"]
        if p["name"] in {"vllm", "torch", "torchvision", "torchaudio", "torchcodec"}
    }
    assert packages["vllm"]["version"] == pin["vllm_version"]
    assert packages["vllm"]["wheels"][0]["hash"] == "sha256:" + pin["vllm_wheel_sha256"]
    assert packages["torch"]["version"] == pin["torch_version"]
    assert packages["torch"]["source"]["registry"] == pin["torch_index"]
    for name in ("torchvision", "torchaudio", "torchcodec"):
        assert packages[name]["version"] == pin[name + "_version"]
        assert packages[name]["source"]["registry"] == pin["torch_index"]
    assert dependency_audit(ROOT / "runtime/uv.lock", allow_local_vllm=True)["status"] == "passed"


def test_allocated_gpu_uuid_uses_visible_device_and_nvml_prefix(monkeypatch):
    from types import SimpleNamespace

    from tdrl.backend import allocated_gpu_uuid

    raw = "26ccf474-1928-483f-869c-ac8ef409ce98"
    for value in (raw, "GPU-" + raw):
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda index, value=value: SimpleNamespace(uuid=value) if index == 0 else None,
        )
        assert allocated_gpu_uuid() == "GPU-" + raw


def test_server_runtime_path_exposes_interpreter_tools_without_venv_activation(
    tmp_path, monkeypatch
):
    import shutil
    import subprocess

    from tdrl.runtime import runtime_path

    runtime_bin = tmp_path / "runtime" / "bin"
    runtime_bin.mkdir(parents=True)
    ninja = runtime_bin / "ninja"
    ninja.write_text("#!/bin/sh\nprintf '1.13.2\\n'\n")
    ninja.chmod(0o755)
    monkeypatch.setattr("sys.executable", str(runtime_bin / "python"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert shutil.which("ninja", path=runtime_path()) == str(ninja)
    assert (
        subprocess.check_output(
            ["ninja", "--version"], env={"PATH": runtime_path()}, text=True
        ).strip()
        == "1.13.2"
    )


def test_port_probe_rejects_live_listener_and_allows_closed_server_time_wait():
    import socket

    from tdrl.runtime import ensure_port_available

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with pytest.raises(OSError):
            ensure_port_available(port)
        with socket.create_connection(("127.0.0.1", port)) as client:
            accepted, _ = listener.accept()
            accepted.close()
            assert client.recv(1) == b""
    ensure_port_available(port)


def test_server_failure_diagnostics_precede_owned_process_termination(tmp_path, monkeypatch):
    import signal
    from types import SimpleNamespace

    import httpx

    from tdrl.runtime import Servers

    servers = Servers(ROOT, tmp_path, {})
    servers.children = [SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout: 0)]
    observed = []

    def terminate(pid, sig):
        report = json.loads((tmp_path / "failure-before-cleanup.json").read_text())
        assert report["exception_type"] == "RemoteProtocolError"
        assert report["owned_api_servers"] == [{"pid": 123, "returncode": None}]
        observed.append((pid, sig))

    monkeypatch.setattr("tdrl.runtime.os.killpg", terminate)
    error = httpx.RemoteProtocolError("Server disconnected without sending a response.")
    servers.__exit__(type(error), error, None)
    assert observed == [(123, signal.SIGTERM)]


def test_servers_resolve_relative_project_root(tmp_path, monkeypatch):
    from tdrl.runtime import Servers

    monkeypatch.chdir(ROOT)
    servers = Servers(Path("."), tmp_path, {})
    assert servers.root == ROOT
    assert (servers.root / "toolchains/cuda-12.9.1/bin/nvcc").is_absolute()


def test_server_compute_reports_worker_allocator_and_whole_gpu_peaks(tmp_path):
    from tdrl.runtime import Servers

    allocator = tmp_path / "allocator"
    allocator.mkdir()
    (allocator / "11.json").write_text(
        json.dumps(
            {
                "pid": 11,
                "peak_allocated_bytes": 40,
                "peak_reserved_bytes": 45,
            }
        )
    )
    (allocator / "12.json").write_text(
        json.dumps(
            {
                "pid": 12,
                "peak_allocated_bytes": 30,
                "peak_reserved_bytes": 35,
            }
        )
    )
    rows = [
        {"total_bytes": 100, "used_bytes": 15, "utilization_percent": 0},
        {"total_bytes": 100, "used_bytes": 83, "utilization_percent": 91},
    ]
    servers = Servers(ROOT, tmp_path, {})
    servers.cold_load_s = 2.5
    servers.monitor = SimpleNamespace(
        rows=rows,
        summary=lambda: {
            "gpu_uuid": "GPU-test",
            "samples": 2,
            "idle_used_bytes": 15,
            "peak_used_bytes": 83,
            "steady_used_bytes": 83,
        },
    )
    compute = servers.compute_metadata()
    assert compute["peak_allocated_bytes"] == 40
    assert compute["gpu_peak_used_bytes"] == 83
    assert compute["gpu_peak_headroom_bytes"] == 17
    assert compute["gpu_peak_used_fraction"] == pytest.approx(0.83)
    assert compute["gpu_peak_utilization_percent"] == 91
    assert [row["pid"] for row in compute["allocator_worker_peaks"]] == [11, 12]


def test_evaluation_namespaces_separate_heads_thresholds_and_budgets():
    from tdrl.comparison import evaluation_contract
    from tdrl.evaluation import dialogue_request_base

    original = {"phase": "development", "thresholds": [1.0, 2.0], "head_sha256": "head-a"}
    variants = [
        (original, "config-a", False),
        ({**original, "thresholds": [1.0, 3.0]}, "config-a", False),
        ({**original, "head_sha256": "head-b"}, "config-a", False),
        ({**original, "budget": "matched"}, "config-a", True),
        (original, "config-b", False),
    ]
    ids = [
        dialogue_request_base(
            evaluation_contract(c, "TDRL", ["goal"], sha, matched), "profile", 17, "same-scenario"
        )
        for c, sha, matched in variants
    ]
    assert len(set(ids)) == len(variants)
    assert "request_namespace" not in original
    assert ids[0] == dialogue_request_base(
        evaluation_contract(original, "TDRL", ["goal"], "config-a", False),
        "profile",
        17,
        "same-scenario",
    )
    assert (
        dialogue_request_base(
            {"phase": "PPDPP-RL", "method": "PPDPP"}, "resistant", 260, "train:en:000243"
        )
        == "PPDPP-RL/PPDPP/resistant/260/train:en:000243"
    )


def test_trial_isolation_preserves_decoding_seeds_and_exact_resume(tmp_path):
    from tdrl.comparison import evaluation_contract

    all_rows = []
    for i, threshold in enumerate([1.0, 2.0]):
        generation, embedding = FakeGeneration(), FakeEmbedding()
        planner = ConditionalPlanner(
            generation, embedding, None, PlannerConfig(0.0, generation_call_cap=1)
        )
        c = evaluation_contract(
            {"phase": "development", "thresholds": [1.0, threshold]},
            "TDRL",
            [],
            "same-config",
            False,
        )
        out = tmp_path / str(i)
        partial = run_dialogue(
            scenario(),
            planner,
            generation,
            embedding,
            "profile",
            17,
            out,
            c,
            turn_budget=2,
            interrupt_after=1,
        )
        result = run_dialogue(
            scenario(), planner, generation, embedding, "profile", 17, out, c, turn_budget=2
        )
        assert result["turns_trace"][0] == partial["turns_trace"][0]
        count = len(generation.requests)
        assert (
            run_dialogue(
                scenario(), planner, generation, embedding, "profile", 17, out, c, turn_budget=2
            )
            == result
        )
        assert len(generation.requests) == count
        all_rows.append([r for t in result["turns_trace"] for r in t["generation"]])
    assert [r["seed"] for r in all_rows[0]] == [r["seed"] for r in all_rows[1]]
    assert {r["request_id"] for r in all_rows[0]}.isdisjoint({r["request_id"] for r in all_rows[1]})


def test_namespaced_journal_isolates_different_prompts_and_counts_actual_calls(
    tmp_path, monkeypatch
):
    from test_core import PIN

    from tdrl.backend import LocalBackend
    from tdrl.comparison import evaluation_contract
    from tdrl.contracts import Decoding
    from tdrl.evaluation import dialogue_request_base
    from tdrl.provenance import Journal

    backend = LocalBackend(
        "http://127.0.0.1:18000/v1",
        PIN,
        "generation",
        Journal(tmp_path / "journal", {"model": "fixture"}),
        tmp_path / "ledger",
    )
    calls = []

    def post(endpoint, payload):
        calls.append(payload)
        return {
            "model": PIN.model_id,
            "choices": [{"message": {"content": "A final response."}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }, 0.1

    monkeypatch.setattr(backend, "_post", post)
    for threshold, prompt in [(1.0, "Hello"), (2.0, "Different history")]:
        c = evaluation_contract(
            {"phase": "development", "thresholds": [threshold, 2]}, "TDRL", [], "config", False
        )
        rid = dialogue_request_base(c, "profile", 17, "same-scenario") + "/turn/0/user"
        messages = [Message("user", prompt)]
        result = backend.generate(messages, "simulator", Decoding(), 17, rid)
        assert backend.generate(messages, "simulator", Decoding(), 17, rid) == result
        with pytest.raises(ValueError, match="different inputs"):
            backend.generate(
                [Message("user", "Unauthorized changed input")], "simulator", Decoding(), 17, rid
            )
    assert len(calls) == 2 and len(backend.requests) == 2
    assert calls[0]["seed"] == calls[1]["seed"] == 17
    assert len(list((tmp_path / "ledger").glob("completed-*.json"))) == 2
