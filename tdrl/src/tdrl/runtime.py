"""Owned local vLLM server lifecycle; scientific execution remains gate-controlled."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import torch

from .audit import dependency_audit
from .backend import MODEL_IDS, LocalBackend, require_h100_slurm, verify_model_cache
from .contracts import Decoding, Message, ModelProvenance
from .evaluation import dialogue_request_base, run_dialogue
from .graph import DirectedGraph, Reference
from .planner import INSTRUCTION, ConditionalPlanner, PlannerConfig
from .provenance import Journal, atomic_json, digest, file_hash
from .reachability import MRNFixed
from .stages import GpuMonitor, Stage, counters
from .state import SERIALIZATION_HASH, DialogueState


def now():
    return datetime.now(timezone.utc).isoformat()


def ensure_port_available(port):
    """Allow closed-server TIME_WAIT sockets, while rejecting every active listener."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
        probe.listen(1)


def runtime_path():
    """Expose tools installed beside this interpreter without activating a shell venv."""
    return str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


def server_command(kind, model_path, model_id, setting):
    """The same explicit launch arguments are parsed at bootstrap and used in Slurm."""
    concurrency = setting.get("request_concurrency", 1)
    if not isinstance(concurrency, int) or concurrency < 1 or concurrency > setting["max_num_seqs"]:
        raise ValueError("request concurrency must fit inside max_num_seqs")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(setting["port"]),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(setting["gpu_memory_utilization"]),
        "--max-model-len",
        str(setting["max_model_len"]),
        "--max-num-seqs",
        str(setting["max_num_seqs"]),
        "--enforce-eager",
        "--no-enable-log-requests",
        "--no-enable-prefix-caching",
    ]
    if kind == "generation":
        command += ["--language-model-only"]
    else:
        command += ["--runner", "pooling", "--convert", "embed"]
    return command


def inference_environment(env):
    """Exclude controller-only policy determinism settings from owned vLLM workers."""
    result = dict(env)
    if result.get("TDRL_POLICY_DETERMINISM") == "strict-backward-v1":
        result.pop("CUBLAS_WORKSPACE_CONFIG", None)
        result.pop("TDRL_POLICY_DETERMINISM", None)
    return result


class Servers:
    def __init__(self, root: Path, output: Path, settings: dict, kinds=("generation", "embedding")):
        # CUDA/FlashInfer launches nested build tools from their own working
        # directories, so project-local toolchain paths must never be relative.
        self.root, self.output, self.settings = root.resolve(), output, settings
        self.children, self.handles = [], []
        self.kinds = kinds

    def __enter__(self):
        start = time.monotonic()
        audit = dependency_audit(self.root / "runtime/uv.lock", allow_local_vllm=True)
        if audit["status"] != "passed":
            raise RuntimeError(f"runtime dependency policy gate blocked: {audit}")
        require_h100_slurm()
        if not (self.root / "toolchains/cuda-12.9.1/bin/nvcc").is_file():
            raise RuntimeError("Pinned project-local CUDA compiler is missing; bootstrap first")
        self.snapshots = verify_model_cache(
            self.root / "configs/models.lock.json", self.root / "models"
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.monitor = GpuMonitor(self.output / "gpu.jsonl")
        self.monitor.__enter__()
        try:
            for kind in self.kinds:
                model_id = MODEL_IDS[kind]
                pin = self.snapshots[model_id]
                setting = self.settings[kind]
                # Never attach to an existing local listener, even with the same model alias.
                ensure_port_available(setting["port"])
                command = server_command(kind, pin["path"], model_id, setting)
                log = (self.output / f"{kind}-server.log").open("a")
                self.handles.append(log)
                env = {
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "VLLM_NO_USAGE_STATS": "1",
                    "DO_NOT_TRACK": "1",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "VLLM_BATCH_INVARIANT": "1" if kind == "embedding" else "0",
                    "CUDA_HOME": str(self.root / "toolchains/cuda-12.9.1"),
                    "FLASHINFER_WORKSPACE_BASE": str((self.output / "kernel-cache").resolve()),
                    "TRITON_CACHE_DIR": str((self.output / "kernel-cache/triton").resolve()),
                    "TORCHINDUCTOR_CACHE_DIR": str(
                        (self.output / "kernel-cache/inductor").resolve()
                    ),
                    "PATH": runtime_path()
                    + os.pathsep
                    + str(self.root / "toolchains/cuda-12.9.1/bin"),
                }
                env = inference_environment(env)
                env["TDRL_ALLOCATOR_TELEMETRY"] = str((self.output / "allocator").resolve())
                env["PYTHONPATH"] = (
                    str(self.root / "instrumentation") + os.pathsep + env.get("PYTHONPATH", "")
                )
                proc = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
                )
                self.children.append(proc)
                atomic_json(
                    self.output / f"{kind}-launch.json",
                    {
                        "command": command,
                        "toolchain_lock_sha256": file_hash(
                            self.root / "configs/cuda-toolchain.lock.json"
                        ),
                        "toolchain_manifest_sha256": file_hash(
                            self.root / "toolchains/cuda-12.9.1/manifest.json"
                        ),
                        "environment": {"CUDA_HOME": env["CUDA_HOME"], "PATH": env["PATH"]},
                        "pid": proc.pid,
                        "started_at": now(),
                        "model": pin,
                        "job_id": os.environ["SLURM_JOB_ID"],
                    },
                )
                self.wait_ready(proc, setting["port"], model_id)
            self.cold_load_s = time.monotonic() - start
            atomic_json(self.output / "cold-load.json", {"seconds": self.cold_load_s})
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    @staticmethod
    def wait_ready(proc, port, model_id):
        deadline = time.monotonic() + 900
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=5) as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("owned vLLM server exited; preserve log and stop gate")
                try:
                    r = client.get(f"http://127.0.0.1:{port}/v1/models")
                    if r.status_code == 200 and model_id in [m["id"] for m in r.json()["data"]]:
                        return
                except httpx.TransportError:
                    pass
                time.sleep(1)
        raise TimeoutError("owned vLLM server readiness timeout")

    def backends(self):
        result = {}
        for kind in self.kinds:
            model_id = MODEL_IDS[kind]
            p = self.snapshots[model_id]
            provenance = ModelProvenance(
                model_id, p["revision"], p["weight_sha256"], p["template_sha256"]
            )
            journal = (
                Journal(self.output / "generation-journal", asdict(provenance))
                if kind == "generation"
                else None
            )
            result[kind] = LocalBackend(
                f"http://127.0.0.1:{self.settings[kind]['port']}/v1",
                provenance,
                kind,
                journal,
                self.output / "request-ledger" / kind,
                self.settings[kind].get("request_concurrency", 1),
            )
        return result

    def compute_metadata(self):
        records = [
            json.loads(p.read_text()) for p in sorted((self.output / "allocator").glob("*.json"))
        ]
        if not records:
            raise ValueError("owned allocator telemetry missing")
        memory = self.monitor.summary()
        total_bytes = max(row["total_bytes"] for row in self.monitor.rows)
        peak_used_bytes = memory["peak_used_bytes"]
        return {
            "cold_load_s": self.cold_load_s,
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in records),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in records),
            "allocator_scope": "largest single owned worker peak; not summed across processes",
            "allocator_worker_peaks": [
                {
                    "pid": record["pid"],
                    "peak_allocated_bytes": record["peak_allocated_bytes"],
                    "peak_reserved_bytes": record["peak_reserved_bytes"],
                }
                for record in records
            ],
            "gpu_peak_used_bytes": peak_used_bytes,
            "gpu_total_bytes": total_bytes,
            "gpu_peak_headroom_bytes": total_bytes - peak_used_bytes,
            "gpu_peak_used_fraction": peak_used_bytes / total_bytes,
            "gpu_peak_utilization_percent": max(
                row["utilization_percent"] for row in self.monitor.rows
            ),
            "gpu_monitor_scope": "whole CUDA-visible GPU sampled by nvidia-smi every 0.5 seconds",
        }

    def __exit__(self, exc_type=None, exc=None, traceback=None):
        if exc is not None:
            atomic_json(
                self.output / "failure-before-cleanup.json",
                {
                    "timestamp": now(),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "owned_api_servers": [
                        {"pid": child.pid, "returncode": child.poll()} for child in self.children
                    ],
                    "note": "Recorded before terminating owned servers; shutdown logs are not root-cause evidence",
                },
            )
        if hasattr(self, "monitor"):
            self.monitor.__exit__()
        # Only processes created by this object; never pkill, scancel, or attach to others.
        for p in self.children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in self.children:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for handle in self.handles:
            handle.close()
        if hasattr(self, "monitor"):
            if self.monitor.rows:
                atomic_json(
                    self.output / "memory.json",
                    {
                        **self.monitor.summary(),
                        "total_bytes": max(row["total_bytes"] for row in self.monitor.rows),
                        "peak_headroom_bytes": max(row["total_bytes"] for row in self.monitor.rows)
                        - self.monitor.summary()["peak_used_bytes"],
                        "peak_used_fraction": self.monitor.summary()["peak_used_bytes"]
                        / max(row["total_bytes"] for row in self.monitor.rows),
                        "scope": "whole CUDA-visible GPU sampled by nvidia-smi every 0.5 seconds",
                    },
                )


def smoke(root: Path, output: Path, judge_config=None):
    """Gate 2 is complete only when every real allocated-H100 assertion passes."""
    settings = json.loads((root / "configs/runtime.json").read_text())
    stage = Stage(
        root,
        output,
        "gate2",
        root / "configs/runtime.json",
        {
            "serialization_sha256": SERIALIZATION_HASH,
            "purpose": "synthetic runtime control; no scientific results",
            "judge_config_sha256": digest(judge_config),
        },
    )
    if stage.reused:
        return
    stage.write("partial")
    started = time.monotonic()
    try:
        gpu = require_h100_slurm()
        with GpuMonitor(output / "allocation-gpu.jsonl") as monitor:
            with Servers(root, output, settings) as servers:
                cold = time.monotonic() - started
                b = servers.backends()
                generation, embedding = b["generation"], b["embedding"]
                from .judge import LocalJudge

                judge = LocalJudge(generation, judge_config) if judge_config else None
                state = DialogueState((Message("user", "I would like to talk about films."),))
                texts = [
                    state.serialize(),
                    DialogueState((Message("user", "Tell me about music."),)).serialize(),
                ]
                first = embedding.embed(texts, INSTRUCTION, "smoke/batch", "smoke/embed")
                again = embedding.embed(texts, INSTRUCTION, "smoke/replay", "smoke/embed-replay")
                torch.save(
                    {"first": first.vectors, "again": again.vectors}, output / "embedding-replay.pt"
                )
                atomic_json(
                    output / "embedding-replay-diagnostics.json",
                    {
                        "exact_equal": torch.equal(first.vectors, again.vectors),
                        "max_absolute_difference": float(
                            (first.vectors - again.vectors).abs().max()
                        ),
                        "purpose": "runtime diagnostic only; not scientific evaluation",
                    },
                )
                # Same input, actual second request; exact equality is required for replay gate.
                if not torch.equal(first.vectors, again.vectors):
                    raise ValueError("live embedding exact replay mismatch")
                reference = Reference(
                    "smoke-only",
                    state,
                    first.vectors[0],
                    "Ask about preferred film genres.",
                    "synthetic-smoke",
                )
                torch.manual_seed(17)
                head = MRNFixed().to(device="cuda", dtype=torch.bfloat16).eval()
                graph = DirectedGraph(
                    [reference], head, 1.0, {"purpose": "synthetic-smoke"}
                ).build()
                simulator = (
                    "Be a skeptical film viewer. Respond naturally, and feel free to refuse."
                )
                decoding = Decoding(max_tokens=128)
                plans = []
                for kind, observation, threshold in [
                    ("covered", state, 0.1),
                    ("off_map", DialogueState((Message("user", "Tell me about music."),)), 0.0),
                ]:
                    planner = ConditionalPlanner(
                        generation,
                        embedding,
                        graph,
                        PlannerConfig(threshold, candidates=2, lookahead=1),
                        decoding=decoding,
                        simulator_decoding=decoding,
                        simulator_prompt=simulator,
                    )
                    plan = planner.act(observation, ["smoke-only"], 17, "smoke/" + kind)
                    if kind == "covered" and (
                        not plan.covered or plan.compute["recovery_triggered"]
                    ):
                        raise ValueError("covered retrieval smoke failed")
                    if kind == "off_map" and (
                        plan.covered
                        or not plan.compute["recovery_triggered"]
                        or len(plan.rollout_scores) != 2
                    ):
                        raise ValueError("conditional off-map candidate/lookahead smoke failed")
                    plans.append(asdict(plan))
                scenario = {
                    "id": "synthetic-smoke",
                    "profile": {"interest": "films"},
                    "personality": "skeptical",
                    "knowledge": [],
                    "target": ["movie", "Synthetic Film"],
                    "messages": [{"role": "user", "content": state.history[0].content}],
                }
                planner = ConditionalPlanner(
                    generation,
                    embedding,
                    graph,
                    PlannerConfig(0.1, candidates=2, lookahead=1),
                    decoding=decoding,
                    simulator_decoding=decoding,
                )
                contract = {
                    "phase": "gate2",
                    "method": "TDRL",
                    "target_nodes": ["smoke-only"],
                    "purpose": "synthetic smoke only",
                }
                partial = run_dialogue(
                    scenario,
                    planner,
                    generation,
                    embedding,
                    "resistant",
                    17,
                    output / "dialogue",
                    contract,
                    turn_budget=3,
                    interrupt_after=1,
                    judge=judge,
                )
                if partial["status"] != "partial":
                    raise ValueError("interruption checkpoint missing")
                resumed = run_dialogue(
                    scenario,
                    planner,
                    generation,
                    embedding,
                    "resistant",
                    17,
                    output / "dialogue",
                    contract,
                    turn_budget=3,
                    judge=judge,
                )
                ids = [r["request_id"] for t in resumed["turns_trace"] for r in t["generation"]]
                if (
                    len(ids) != len(set(ids))
                    or resumed["turns_trace"][0] != partial["turns_trace"][0]
                ):
                    raise ValueError("resume duplicated or altered completed turn")
                count = len(generation.requests)
                replay = run_dialogue(
                    scenario,
                    planner,
                    generation,
                    embedding,
                    "resistant",
                    17,
                    output / "dialogue",
                    contract,
                    turn_budget=3,
                    judge=judge,
                )
                if replay != resumed or len(generation.requests) != count:
                    raise ValueError("completed dialogue exact replay failed")
                request = [Message("user", "Say hello in one sentence.")]
                one = generation.generate(
                    request, "assistant", decoding, 17, "smoke/generation-replay"
                )
                two = generation.generate(
                    request, "assistant", decoding, 17, "smoke/generation-replay"
                )
                if one != two:
                    raise ValueError("generation journal replay mismatch")
                namespace_records = []
                before = len(generation.requests)
                for trial, text in [("threshold-a", "Say hello."), ("threshold-b", "Say goodbye.")]:
                    scoped = {"phase": "development", "method": "TDRL", "request_namespace": trial}
                    request_id = (
                        dialogue_request_base(scoped, "profile", 17, "same-smoke-scenario")
                        + "/turn/0/user"
                    )
                    messages = [Message("user", text)]
                    actual = generation.generate(messages, "simulator", decoding, 17, request_id)
                    replayed = generation.generate(messages, "simulator", decoding, 17, request_id)
                    if actual != replayed:
                        raise ValueError("namespaced generation replay mismatch")
                    namespace_records.append({"request_id": request_id, "seed": actual.seed})
                if (
                    len(generation.requests) - before != 2
                    or len({r["request_id"] for r in namespace_records}) != 2
                ):
                    raise ValueError("distinct trials did not execute separately")
                atomic_json(
                    output / "namespace-smoke.json",
                    {
                        "status": "passed",
                        "scientific_result": False,
                        "distinct_trial_requests": 2,
                        "cache_replays": 2,
                        "shared_scenario_seed": True,
                        "records": namespace_records,
                    },
                )
                atomic_json(
                    output / "checks.json",
                    {
                        "status": "passed",
                        "scientific_result": False,
                        "dual_BF16_residency": True,
                        "real_embedding_shape": list(first.vectors.shape),
                        "live_embedding_exact_replay": True,
                        "non_thinking_final_text": True,
                        "covered_and_conditional_off_map": True,
                        "complete_dialogue": True,
                        "checkpoint_boundary_interruption_resume": True,
                        "completed_exact_replay": True,
                        "generation_exact_replay": True,
                        "trial_namespace_isolation_and_replay": True,
                        "plans": plans,
                        "transport": {
                            "endpoints": [generation.url, embedding.url],
                            "trust_env": False,
                            "redirects": False,
                            "offline_models": True,
                            "scope": "application requests and owned launch configuration; not OS egress enforcement",
                        },
                    },
                )
                compute = counters(generation, embedding)
            memory = monitor.summary()
            resident_memory = json.loads((output / "memory.json").read_text())
            memory["steady_used_bytes"] = resident_memory["steady_used_bytes"]
            memory["steady_scope"] = "dual-resident before owned server shutdown"
        allocator = [json.loads(p.read_text()) for p in (output / "allocator").glob("*.json")]
        if not allocator:
            raise ValueError("owned vLLM allocator telemetry absent")
        compute.update(
            cold_load_s=cold,
            h100_seconds=time.monotonic() - started,
            peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in allocator),
            peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in allocator),
        )
        stage.write(
            "complete",
            {
                "gpu": {**gpu, "memory": memory},
                "models": {
                    "pins": json.loads((root / "configs/models.lock.json").read_text()),
                    "verified": servers.snapshots,
                },
                "seeds": {"smoke": 17},
            },
            compute,
        )
    except BaseException as error:
        stage.write("failed", error=error)
        raise
