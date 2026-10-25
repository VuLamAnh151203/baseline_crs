#!/usr/bin/env python3
"""Pause owned queue, await durable workers, apply committed code and gated restart."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--job-id", required=True)
    a = p.parse_args()
    root = a.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root / "scripts"))
    from run_in_allocation import validate_allocation

    validate_allocation(
        subprocess.check_output(["scontrol", "show", "job", "-o", a.job_id], text=True),
        os.getuid(),
        gpu_count=2,
    )
    shared = root / ("artifacts/two-gpu-" + a.job_id)
    output = root / ("artifacts/branch-execution-" + a.job_id)
    output.mkdir(exist_ok=True)
    status = output / "transition-status.json"

    def progress(phase, **extra):
        save(
            status,
            {
                "schema_version": 1,
                "status": "running",
                "phase": phase,
                "time": time.time(),
                **extra,
            },
        )

    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise ValueError("transition needs committed clean source")
        if (shared / "selection/selection.lock").exists() or (shared / "test").exists():
            raise ValueError("branch transition is authorized before global lock/test only")
        pid = int((root / f"logs/two-gpu-{a.job_id}-critic-concurrent-driver.pid").read_text())
        proc = Path(f"/proc/{pid}")
        cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if (
            proc.stat().st_uid != os.getuid()
            or (proc / "cwd").resolve() != root
            or "scripts/run_two_gpu.py" not in cmd
            or f"--job-id {a.job_id}" not in cmd
        ):
            raise ValueError("refuse to stop unrecognized controller")
        save(output / "source-queue.json", json.loads((shared / "queue-status.json").read_text()))
        os.kill(pid, signal.SIGTERM)  # Only owned controller; existing Slurm steps finish durably.
        marker = shared / "tasks/policy-DuRecDial2.0/stop.requested"
        save(
            shared / "queue-status.json",
            {
                "schema_version": 1,
                "status": "partial",
                "error": "parallel candidate transition: waiting for selection boundary; RL continues",
            },
        )
        progress("await_selection_boundary", source_commit=source_commit)
        deadline = time.monotonic() + 6 * 3600
        selection_status = shared / "tasks/selection-TopDial/task-status.json"
        while (
            not selection_status.exists()
            or json.loads(selection_status.read_text())["status"] != "complete"
        ):
            if (
                selection_status.exists()
                and json.loads(selection_status.read_text())["status"] == "partial"
            ):
                raise ValueError("original selection failed; retain all evidence")
            if time.monotonic() > deadline:
                raise TimeoutError("selection did not reach its durable stage boundary")
            time.sleep(10)
        marker = shared / "tasks/policy-DuRecDial2.0/stop.requested"
        if marker.is_symlink():
            marker.unlink()
        if marker.exists():
            raise ValueError("unknown pre-existing worker stop marker")
        marker.write_text("user-authorized parallel candidate transition\n")
        progress("await_durable_workers", source_commit=source_commit)
        while True:
            steps = subprocess.check_output(
                ["squeue", "-h", "--steps", "-j", a.job_id, "-o", "%i"], text=True
            ).splitlines()
            if not [s for s in steps if not s.endswith(".batch") and not s.endswith(".extern")]:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("owned workers did not reach durable boundary")
            time.sleep(10)
        rl = shared / "tasks/policy-DuRecDial2.0/PPDPP-RL/DuRecDial2.0"
        if list((shared / "tasks/policy-DuRecDial2.0").rglob("pending-*.json")):
            raise ValueError("RL has unresolved inference outcome")
        for ds in ["TopDial", "DuRecDial2.0"]:
            task = shared / f"tasks/selection-{ds}"
            if json.loads((task / "task-status.json").read_text())["status"] != "complete" or list(
                task.rglob("pending-*.json")
            ):
                raise ValueError(
                    "original selection did not complete cleanly; retain all artifacts"
                )
        snapshot = output / "source-RL"
        snapshot.mkdir()
        for name in ["manifest.json", "contract.json", "progress.json", "policy-replay.json"]:
            (snapshot / name).write_bytes((rl / name).read_bytes())
        progress("apply_committed_execution", source_commit=source_commit)
        subprocess.run(["git", "fetch", str(a.bundle), "HEAD"], check=True)
        if (
            subprocess.check_output(["git", "rev-parse", "FETCH_HEAD"], text=True).strip()
            != a.commit
        ):
            raise ValueError("bundle commit mismatch")
        subprocess.run(["git", "merge", "--ff-only", "FETCH_HEAD"], check=True)
        env = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            VLLM_NO_USAGE_STATS="1",
            DO_NOT_TRACK="1",
            OMP_NUM_THREADS="4",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            TDRL_POLICY_DETERMINISM="strict-backward-v1",
            TDRL_CRITIC_WORKERS="10",
            TDRL_CANDIDATE_WORKERS="3",
        )
        progress("both_gpu_smokes")
        processes, handles = [], []
        for slot in [0, 1]:
            handle = (root / f"logs/branch-execution-{a.job_id}-smoke-{slot}.log").open("a")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    [
                        "srun",
                        "--jobid=" + a.job_id,
                        "--ntasks=1",
                        "--cpus-per-task=8",
                        "--gres=gpu:nvidia_h100_80gb_hbm3:1",
                        "--exclusive",
                        "--exact",
                        "runtime/.venv/bin/python",
                        "scripts/smoke_branch_parallel.py",
                        "--slot",
                        str(slot),
                        "--shared",
                        str(shared),
                        "--output",
                        str(output / f"smoke-{slot}"),
                        "--checkpoint",
                        str(rl / "checkpoint.pt"),
                    ],
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [p.wait() for p in processes]
        for handle in handles:
            handle.close()
        if codes != [0, 0]:
            raise ValueError("branch smoke failed: " + str(codes))
        checks = [json.loads((output / f"smoke-{i}/checks.json").read_text()) for i in [0, 1]]
        if (
            len({c["gpu"]["uuid"] for c in checks}) != 2
            or len({c["checkpoint_sha256"] for c in checks}) != 1
        ):
            raise ValueError("branch smoke GPU/checkpoint mismatch")
        inherited = root / f"artifacts/critic-execution-{a.job_id}/amendment.json"
        amendment = output / "amendment.json"
        save(
            amendment,
            {
                "schema_version": 1,
                "status": "passed",
                "authorization": "user 2026-10-09: parallel three Full candidate branches",
                "commit": a.commit,
                "source_commit": source_commit,
                "allocation_id": a.job_id,
                "source_updates": json.loads((rl / "progress.json").read_text())["episode"] + 1,
                "source_checkpoint_sha256": checks[0]["checkpoint_sha256"],
                "critic_amendment": {"path": str(inherited), "sha256": sha(inherited)},
                "smokes": [
                    {
                        "path": str(output / f"smoke-{i}/checks.json"),
                        "sha256": sha(output / f"smoke-{i}/checks.json"),
                    }
                    for i in [0, 1]
                ],
                "source_manifest_sha256": sha(snapshot / "manifest.json"),
                "fresh_paired_selection": True,
                "batch_dependent_generation": True,
            },
        )
        env["TDRL_BRANCH_AMENDMENT"] = str(amendment)
        subprocess.run(
            [
                "runtime/.venv/bin/python",
                "-c",
                "from pathlib import Path; from tdrl.branch_execution import branch_evidence; print(branch_evidence(Path.cwd()))",
            ],
            env=env,
            check=True,
        )
        marker.rename(output / "source-stop.requested")
        progress("restart_queue")
        with (root / f"logs/two-gpu-{a.job_id}-branch-concurrent-driver.log").open("a") as handle:
            driver = subprocess.Popen(
                [
                    "runtime/.venv/bin/python",
                    "scripts/run_two_gpu.py",
                    "--job-id",
                    a.job_id,
                    "--source",
                    "artifacts/ppdpp-tdrl-concurrent-v2-57711",
                    "--output",
                    str(shared),
                ],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (root / f"logs/two-gpu-{a.job_id}-branch-concurrent-driver.pid").write_text(
            str(driver.pid) + "\n"
        )
        save(
            status,
            {
                "schema_version": 1,
                "status": "complete",
                "phase": "queue_restarted",
                "driver_pid": driver.pid,
                "amendment": str(amendment),
                "time": time.time(),
            },
        )
    except Exception as error:
        save(
            status,
            {"schema_version": 1, "status": "partial", "error": str(error), "time": time.time()},
        )
        raise


if __name__ == "__main__":
    main()
