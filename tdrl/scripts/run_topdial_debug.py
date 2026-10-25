#!/usr/bin/env python3
"""Two-GPU development-only planner diagnostics with a both-GPU smoke barrier."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from run_in_allocation import validate_allocation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path("configs/topdial-debug-v2.json"))
parser.add_argument("--output-prefix", default="topdial-debug-v2")
args = parser.parse_args()
root = Path.cwd()
config = json.loads((root / args.config).read_text())
evaluation_slots = config.get("evaluation_slots", [0, 1])
if evaluation_slots not in ([0], [0, 1]):
    raise ValueError("unknown evaluation GPU slots")
job = (root / "logs/active-allocation-job-id.txt").read_text().strip()
validate_allocation(
    subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True),
    os.getuid(),
    gpu_count=2,
)
steps = subprocess.check_output(
    ["squeue", "-h", "--steps", "-j", job, "-o", "%i"], text=True
).splitlines()
if any(not x.endswith((".batch", ".extern")) for x in steps):
    raise ValueError("allocation has active steps; do not attach to unknown work")
if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
    raise ValueError("diagnostics require clean committed source")
output = root / ("artifacts/" + args.output_prefix + "-" + job)
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
    TDRL_CRITIC_WORKERS="10",
)
status = output / "queue-status.json"
try:
    for mode in ["smoke", "evaluate"]:
        status.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "running",
                    "mode": mode,
                    "job_id": job,
                    "time": time.time(),
                }
            )
            + "\n"
        )
        processes, handles = [], []
        slots = [0, 1] if mode == "smoke" else evaluation_slots
        for slot in slots:
            handle = (root / f"logs/{args.output_prefix}-{job}-{mode}-{slot}.log").open("a")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    [
                        "srun",
                        "--jobid=" + job,
                        "--ntasks=1",
                        "--cpus-per-task=8",
                        "--gres=gpu:nvidia_h100_80gb_hbm3:1",
                        "--exclusive",
                        "--exact",
                        "runtime/.venv/bin/python",
                        "scripts/debug_topdial_planner.py",
                        "--config",
                        str(args.config),
                        "--slot",
                        str(slot),
                        "--mode",
                        mode,
                        "--output",
                        str(output / (f"smoke-{slot}" if mode == "smoke" else f"eval-{slot}")),
                    ],
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [p.wait() for p in processes]
        for h in handles:
            h.close()
        if codes != [0] * len(slots):
            raise ValueError(f"{mode} failed: {codes}; evidence retained")
        if mode == "smoke":
            checks = [json.loads((output / f"smoke-{i}/checks.json").read_text()) for i in [0, 1]]
            if len({x["gpu"]["uuid"] for x in checks}) != 2 or any(
                x["status"] != "passed" for x in checks
            ):
                raise ValueError("both distinct-GPU runtime gates required")
    subprocess.run(
        [
            "runtime/.venv/bin/python",
            "scripts/summarize_topdial_debug.py",
            str(output),
            "--mrn-split"
            if config.get("head_types") == ["MRNFixed"] and evaluation_slots == [0, 1]
            else "--mrn-only"
            if evaluation_slots == [0]
            else "--numeric-baseline"
            if config.get("head_source")
            else "--standard",
        ],
        env=env,
        check=True,
    )
    status.write_text(
        json.dumps({"schema_version": 1, "status": "complete", "job_id": job, "time": time.time()})
        + "\n"
    )
except Exception as error:
    status.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "partial",
                "error": str(error),
                "job_id": job,
                "time": time.time(),
            }
        )
        + "\n"
    )
    raise
