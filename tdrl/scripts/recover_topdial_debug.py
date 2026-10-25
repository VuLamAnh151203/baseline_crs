#!/usr/bin/env python3
"""Retain 640 complete dialogues; finish isolated IQE plus independent head refit."""

import json
import os
import subprocess
from pathlib import Path

from run_in_allocation import validate_allocation

root = Path.cwd()
job = (root / "logs/active-allocation-job-id.txt").read_text().strip()
validate_allocation(
    subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True),
    os.getuid(),
    gpu_count=2,
)
if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
    raise ValueError("committed clean source required")
steps = subprocess.check_output(
    ["squeue", "-h", "--steps", "-j", job, "-o", "%i"], text=True
).splitlines()
if any(not s.endswith((".batch", ".extern")) for s in steps):
    raise ValueError("unknown active steps; refuse reuse")
original = root / f"artifacts/topdial-debug-v2-{job}"
if list((original / "eval-1").rglob("pending-*.json")):
    raise ValueError("unresolved original requests")
output = root / f"artifacts/topdial-debug-v2-namespace-{job}"
output.mkdir(exist_ok=True)
env = dict(
    os.environ,
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    VLLM_NO_USAGE_STATS="1",
    DO_NOT_TRACK="1",
    OMP_NUM_THREADS="4",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
    TDRL_POLICY_DETERMINISM="strict-backward-v1",
    TDRL_CANDIDATE_WORKERS="3",
)
status = output / "queue-status.json"
handles = []


def launch(name, args):
    h = (root / f"logs/topdial-namespace-{job}-{name}.log").open("a")
    handles.append(h)
    return subprocess.Popen(
        [
            "srun",
            "--jobid=" + job,
            "--ntasks=1",
            "--cpus-per-task=8",
            "--gres=gpu:nvidia_h100_80gb_hbm3:1",
            "--exclusive",
            "--exact",
            "runtime/.venv/bin/python",
            *args,
        ],
        env=env,
        stdout=h,
        stderr=subprocess.STDOUT,
    )


try:
    status.write_text(
        json.dumps(
            {"schema_version": 1, "status": "running", "phase": "fresh_both_gpu_namespace_smokes"}
        )
        + "\n"
    )
    smoke = [
        launch(
            "smoke-" + str(i),
            [
                "scripts/debug_topdial_planner.py",
                "--slot",
                str(i),
                "--mode",
                "smoke",
                "--output",
                str(output / f"smoke-{i}"),
            ],
        )
        for i in [0, 1]
    ]
    if [p.wait() for p in smoke] != [0, 0]:
        raise ValueError("namespace smoke failed")
    checks = [json.loads((output / f"smoke-{i}/checks.json").read_text()) for i in [0, 1]]
    if len({c["gpu"]["uuid"] for c in checks}) != 2:
        raise ValueError("distinct GPU gate required")
    status.write_text(
        json.dumps(
            {"schema_version": 1, "status": "running", "phase": "remaining_IQE_and_numeric_refit"}
        )
        + "\n"
    )
    iqe = launch(
        "IQE-evaluation",
        [
            "scripts/debug_topdial_planner.py",
            "--slot",
            "1",
            "--mode",
            "evaluate",
            "--output",
            str(output / "eval-iqe"),
            "--variants-only",
            "repair-IQE-R",
            "repair-IQE-Full",
        ],
    )
    codes = []
    for head in ["MRNFixed", "IQE"]:
        p = launch(
            "numeric-" + head,
            [
                "scripts/train_head_numerics.py",
                "--head",
                head,
                "--output",
                str(root / f"artifacts/head-numerics-v2-{job}" / head),
            ],
        )
        codes.append(p.wait())
    if iqe.wait() != 0 or codes != [0, 0]:
        raise ValueError("IQE continuation/numeric fitting failed; preserve evidence")
    subprocess.run(
        [
            "runtime/.venv/bin/python",
            "scripts/summarize_topdial_debug.py",
            str(original),
            str(output / "eval-iqe"),
        ],
        env=env,
        check=True,
    )
    status.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "phase": "v2_report_complete_numeric_heads_require_dialogue_validation",
            }
        )
        + "\n"
    )
except Exception as e:
    status.write_text(
        json.dumps({"schema_version": 1, "status": "partial", "error": str(e)}) + "\n"
    )
    raise
finally:
    for h in handles:
        h.close()
