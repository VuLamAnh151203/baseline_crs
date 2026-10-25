#!/usr/bin/env python3
"""Run later tasks in the registered owned allocation, without creating another job."""

import argparse
import json
import os
import re
import shlex
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DEFAULT_REMOTE_BASE = "/home/u26anhvl/crs"


def remote_project_root():
    base = os.environ.get("TDRL_REMOTE_ROOT", DEFAULT_REMOTE_BASE)
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", base) or ".." in PurePosixPath(base).parts:
        raise ValueError("TDRL_REMOTE_ROOT must be a normalized absolute path")
    return PurePosixPath(base) / "tdrl"


def validate_allocation(record, uid, gpu_count=1):
    fields = dict(token.split("=", 1) for token in record.split() if "=" in token)
    if fields.get("UserId", "").split("(")[-1].rstrip(")") != str(uid):
        raise ValueError("allocation owner mismatch")
    if fields.get("JobState") != "RUNNING":
        raise ValueError(
            "registered allocation is not RUNNING; wait for Slurm or inspect its state"
        )
    if fields.get("JobName") != "tdrl-persistent-allocation":
        raise ValueError("registered job is not the owned persistent allocation")
    if fields.get("NumCPUs") != "16" or fields.get("TimeLimit") != "20-00:00:00":
        raise ValueError("allocation CPU or wall-time contract mismatch")
    resources = dict(
        part.split("=", 1) for part in fields.get("AllocTRES", "").split(",") if "=" in part
    )
    if (
        gpu_count not in {1, 2}
        or resources.get("mem") != "128G"
        or resources.get("gres/gpu") != str(gpu_count)
    ):
        raise ValueError("allocation RAM or GPU contract mismatch")
    if fields.get("TresPerNode") != "gres/gpu:nvidia_h100_80gb_hbm3:" + str(gpu_count):
        raise ValueError("allocation must contain a full H100 80 GB")


def step_command(job_id, command, worker_cpus=16):
    if not job_id.isdecimal() or not command:
        raise ValueError("numeric allocation ID and explicit command are required")
    return [
        "srun",
        "--jobid=" + job_id,
        "--ntasks=1",
        "--cpus-per-task=" + str(worker_cpus),
        "--gres=gpu:nvidia_h100_80gb_hbm3:1",
        *command,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-execute", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    if not args.remote_execute:
        root = remote_project_root()
        invocation = [
            "env",
            "TDRL_REMOTE_ROOT=" + str(root.parent),
            "python3",
            str(root / "scripts/run_in_allocation.py"),
            "--remote-execute",
        ]
        if args.job_id:
            invocation += ["--job-id", args.job_id]
        invocation += ["--", *command]
        return subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=15",
                os.environ.get("TDRL_SSH_ALIAS", "H100"),
                shlex.join(invocation),
            ],
            check=False,
        ).returncode
    root = remote_project_root()
    os.chdir(root)
    registered = (root / "logs/active-allocation-job-id.txt").read_text().strip()
    job_id = args.job_id or registered
    if job_id != registered:
        raise ValueError("task must run in the registered active allocation")
    record = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True)
    validate_allocation(record, os.getuid())
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ValueError("commit scientific source before running an allocation step")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
        OMP_NUM_THREADS="8",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        TDRL_POLICY_DETERMINISM="strict-backward-v1",
    )
    output = root / "artifacts" / f"recovery-{job_id}" / "allocation-steps"
    output.mkdir(parents=True, exist_ok=True)
    task_id = uuid.uuid4().hex
    metadata = {
        "schema_version": 1,
        "allocation_job_id": job_id,
        "task_id": task_id,
        "command": command,
        "submitted_code_commit": commit,
        "status": "queued",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output / (task_id + ".json")
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    # Reserve all 16 CPUs and the one GPU; do not use --overlap or create another sbatch job.
    result = subprocess.run(step_command(job_id, command), env=environment, check=False)
    metadata.update(
        status="complete" if result.returncode == 0 else "failed",
        exit_code=result.returncode,
        ended_at=datetime.now(timezone.utc).isoformat(),
    )
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
