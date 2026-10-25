#!/usr/bin/env python3
"""Run a bounded dependency queue in the registered two-H100 allocation."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from run_in_allocation import validate_allocation


def task_plan():
    boot = {"bootstrap-0", "bootstrap-1"}
    plan = [
        ("bootstrap-0", "bootstrap", None, None, set()),
        ("bootstrap-1", "bootstrap", None, None, set()),
        ("policy-DuRecDial2.0", "policy", "DuRecDial2.0", None, boot),
        ("offline-DuRecDial2.0", "offline", "DuRecDial2.0", None, boot),
        ("policy-TopDial", "policy", "TopDial", None, boot),
        (
            "selection-DuRecDial2.0",
            "select-shard",
            "DuRecDial2.0",
            None,
            boot | {"offline-DuRecDial2.0"},
        ),
        ("selection-TopDial", "select-shard", "TopDial", None, boot | {"offline-DuRecDial2.0"}),
        ("policy-dev-TopDial", "policy-dev", "TopDial", None, {"policy-TopDial"}),
        ("policy-dev-DuRecDial2.0", "policy-dev", "DuRecDial2.0", None, {"policy-DuRecDial2.0"}),
        (
            "lock",
            "lock",
            None,
            None,
            {
                "policy-TopDial",
                "policy-DuRecDial2.0",
                "selection-TopDial",
                "selection-DuRecDial2.0",
                "policy-dev-TopDial",
                "policy-dev-DuRecDial2.0",
            },
        ),
        ("test-policy", "test", None, "policy", {"lock"}),
        ("test-full", "test", None, "full", {"lock"}),
        ("report", "report", None, None, {"test-policy", "test-full"}),
    ]

    if os.environ.get("TDRL_CANDIDATE_WORKERS") == "3":
        rename = {
            "selection-" + ds: "selection-branches-" + ds for ds in ["TopDial", "DuRecDial2.0"]
        }
        plan = [
            (rename.get(n, n), m, d, r, {rename.get(dep, dep) for dep in deps})
            for n, m, d, r, deps in plan
        ]
    return plan


def eligible_tasks(plan, done, running):
    return [task for task in plan if task[0] not in done | running and task[4] <= done]


def link(destination, target):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != target.resolve():
            raise ValueError("assembled artifact location mismatch")
    else:
        destination.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def assemble_completed_fit(shared, name):
    output = shared / "tasks" / name
    if name == "offline-DuRecDial2.0":
        for prefix in ["embeddings", "learning"]:
            link(shared / prefix / "DuRecDial2.0", output / prefix / "DuRecDial2.0")
    if name.startswith("policy-") and not name.startswith("policy-dev-"):
        dataset = name.removeprefix("policy-")
        for prefix in ["PPDPP-SFT", "PPDPP-RL"]:
            link(shared / prefix / dataset, output / prefix / dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    record = subprocess.check_output(["scontrol", "show", "job", "-o", args.job_id], text=True)
    validate_allocation(record, os.getuid(), gpu_count=2)
    if (root / "logs/active-allocation-job-id.txt").read_text().strip() != args.job_id:
        raise ValueError("unregistered two-GPU allocation")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("two-GPU tasks require committed source")
    shared, source = args.output.resolve(), args.source.resolve()
    shared.mkdir(parents=True, exist_ok=True)
    holder = root / f"artifacts/allocation-{args.job_id}"
    for name in ["allocation.json", "slurm-job.txt"]:
        link(shared / name, holder / name)
    link(shared / "stop.requested", holder / "stop.requested")
    for name in ["embeddings/TopDial", "learning/TopDial", "PPDPP-SFT/TopDial"]:
        link(shared / name, source / name)
    environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
        OMP_NUM_THREADS="4",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        TDRL_POLICY_DETERMINISM="strict-backward-v1",
        TDRL_ALLOCATION_GPU_COUNT="2",
    )
    plan, done, running = task_plan(), set(), {}
    for task in plan:
        status = shared / "tasks" / task[0] / "task-status.json"
        if status.exists() and json.loads(status.read_text())["status"] == "complete":
            done.add(task[0])
            assemble_completed_fit(shared, task[0])
    handles = {}
    try:
        while len(done) < len(plan):
            if (holder / "stop.requested").exists():
                raise InterruptedError("allocation wall-limit stop requested")
            for slot, (name, process) in list(running.items()):
                rc = process.poll()
                if rc is None:
                    continue
                handles.pop(name).close()
                del running[slot]
                if rc:
                    raise RuntimeError(
                        f"owned task {name} failed with exit {rc}; artifacts retained"
                    )
                done.add(name)
                assemble_completed_fit(shared, name)
                if {"bootstrap-0", "bootstrap-1"} <= done:
                    evidence = []
                    for i in range(2):
                        bootstrap = shared / "tasks" / f"bootstrap-{i}"
                        evidence.append(
                            {
                                "gpu": json.loads((bootstrap / "task-gpu.json").read_text()),
                                "gate2_sha256": hashlib.sha256(
                                    (bootstrap / "gate2/manifest.json").read_bytes()
                                ).hexdigest(),
                            }
                        )
                    if len({r["gpu"]["uuid"] for r in evidence}) != 2:
                        raise ValueError("bootstrap tasks used the same physical GPU")
                    (shared / "both-gpu-smoke.json").write_text(
                        json.dumps(evidence, indent=2) + "\n"
                    )
                    for gate in ["gate0", "gate1", "gate2"]:
                        link(shared / gate, shared / "tasks/bootstrap-0" / gate)
            ready = eligible_tasks(plan, done, {n for n, _ in running.values()})
            for slot in range(2):
                if slot in running or not ready:
                    continue
                # Bootstrap rank identity and port identity must match.
                task = next(
                    (
                        t
                        for t in ready
                        if not t[0].startswith("bootstrap-") or t[0] == f"bootstrap-{slot}"
                    ),
                    None,
                )
                if task is None:
                    continue
                ready.remove(task)
                name, mode, dataset, role, _ = task
                output = shared / "tasks" / name
                command = [
                    "srun",
                    "--jobid=" + args.job_id,
                    "--ntasks=1",
                    "--cpus-per-task=8",
                    "--gres=gpu:nvidia_h100_80gb_hbm3:1",
                    "--exclusive",
                    "--exact",
                    "runtime/.venv/bin/python",
                    "-m",
                    "tdrl",
                    "two-gpu-task",
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    "--shared",
                    str(shared),
                    "--slot",
                    str(slot),
                    "--mode",
                    mode,
                ]
                if dataset:
                    command += ["--dataset", dataset]
                if role:
                    command += ["--role", role]
                log = root / "logs" / f"two-gpu-{args.job_id}-{name}.log"
                handles[name] = log.open("a")
                running[slot] = (
                    name,
                    subprocess.Popen(
                        command, env=environment, stdout=handles[name], stderr=subprocess.STDOUT
                    ),
                )
            (shared / "queue-status.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "job_id": args.job_id,
                        "done": sorted(done),
                        "running": {str(k): n for k, (n, _) in running.items()},
                        "status": "running",
                    }
                )
                + "\n"
            )
            time.sleep(1)
        (shared / "queue-status.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job_id": args.job_id,
                    "status": "complete",
                    "done": sorted(done),
                }
            )
            + "\n"
        )
    except Exception as error:
        # Stop sibling workers at their own durable boundaries; do not release allocation.
        for name, _ in running.values():
            marker = shared / "tasks" / name / "stop.requested"
            if marker.is_symlink():
                marker.unlink()
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("two-GPU queue failed; retain durable state\n")
        (shared / "queue-status.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "partial",
                    "error": str(error),
                    "done": sorted(done),
                }
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
