#!/usr/bin/env python3
"""Real H100 serial/concurrent critic benchmark, no optimizer updates or test resources."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from tdrl.backend import LocalBackend, require_h100_slurm
from tdrl.contracts import Message
from tdrl.fidelity import replay_policy
from tdrl.judge import LocalJudge
from tdrl.policy_optimizer import policy_optimizer
from tdrl.policy_training import load_policy
from tdrl.provenance import Journal, atomic_json, file_hash
from tdrl.runtime import Servers
from tdrl.state import DialogueState
from tdrl.training import load_prepared

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--slot", type=int, choices=[0, 1], required=True)
p.add_argument("--checkpoint", type=Path, required=True)
a = p.parse_args()
root = Path.cwd()
cfg = json.loads((root / "configs/comparison-judge-v2-sample500-seed17.json").read_text())
settings = json.loads((root / "configs/runtime.json").read_text())
for kind in ["generation", "embedding"]:
    settings[kind]["port"] += a.slot * 100
policy, tokenizer, _ = load_policy(root, a.checkpoint)
optimizer = policy_optimizer(policy, cfg["ppdpp"], "rl")
optimizer.reserve()
controller = replay_policy(policy, tokenizer, load_prepared(root, "DuRecDial2.0", "train")[0])
scenario = {"target": ["Movie recommendation", "The Star"], "profile": {}, "knowledge": []}
state = DialogueState(
    (
        Message("assistant", "Would you like to watch The Star?"),
        Message("user", "I'll watch The Star tonight."),
    )
)
measurements = {}
with Servers(root, a.output / "servers", settings) as servers:
    b = servers.backends()
    for workers in [1, 10]:
        directory = a.output / str(workers)
        backend = LocalBackend(
            b["generation"].url,
            b["generation"].model,
            "generation",
            Journal(directory / "journal", asdict(b["generation"].model)),
            directory / "ledger",
        )
        judge = LocalJudge(backend, cfg["judge"])
        os.environ["TDRL_CRITIC_WORKERS"] = str(workers)
        start = time.monotonic()
        result = judge.feedback(state, scenario, 17, "critic-execution/same-seeds")
        wall = time.monotonic() - start
        count = len(backend.requests)
        replay = judge.feedback(state, scenario, 17, "critic-execution/same-seeds")
        if replay != result or len(backend.requests) != count or count != 10:
            raise ValueError("critic exact replay/request count failed")
        if [r.request_id for r in backend.consumed_requests[10:]] != [
            f"critic-execution/same-seeds/{i}" for i in range(10)
        ] and workers == 1:
            raise ValueError("serial fixture IDs changed")
        atomic_json(directory / "feedback.json", result)
        measurements[str(workers)] = {
            "wall_s": wall,
            "reward": result["reward"],
            "requests": count,
            "seeds": sorted(r.seed for r in backend.requests),
            "ids": sorted(r.request_id for r in backend.requests),
        }
        backend.client.close()
    if (
        measurements["1"]["seeds"] != measurements["10"]["seeds"]
        or measurements["1"]["ids"] != measurements["10"]["ids"]
    ):
        raise ValueError("serial/concurrent request identity changed")
    b["embedding"].embed(
        [state.serialize()],
        "Represent the dialogue state.",
        "critic-execution",
        "critic-execution/live",
    )
    compute = servers.compute_metadata()
pending = list(a.output.rglob("pending-*.json"))
if pending:
    raise ValueError("unresolved critic smoke request")
atomic_json(
    a.output / "checks.json",
    {
        "schema_version": 1,
        "status": "passed",
        "gpu": require_h100_slurm(),
        "requests": 10,
        "exact_journal_replay": True,
        "pending_requests": 0,
        "optimizer_updates": 0,
        "controller_replay": controller,
        "checkpoint_sha256": file_hash(a.checkpoint),
        "measurements": measurements,
        "compute": compute,
        "serial_concurrent_text_equality_claim": False,
    },
)
