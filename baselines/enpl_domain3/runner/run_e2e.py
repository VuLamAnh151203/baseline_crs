#!/usr/bin/env python3
"""Prepare EnPL, run dev/lock/smoke, and optionally the fixed-400 test."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tdrl.baseline_test_setting import load_fixed_test_setting
from tdrl.baselines.enpl.bank import load_bank
from tdrl.baselines.enpl.selection import verify_reference_lock, verify_selection
from tdrl.provenance import atomic_json, digest, file_hash

DATASETS = ("TopDial", "DuRecDial2.0")
PORT_START, PORT_STOP = 60_000, 65_000


def now():
    return datetime.now(timezone.utc).isoformat()


def ports(job_id):
    blocks = (PORT_STOP - PORT_START) // 4
    initial = int(job_id) % blocks
    for offset in range(blocks):
        base = PORT_START + ((initial + offset) % blocks) * 4
        values = tuple(base + index for index in range(4))
        probes = []
        try:
            for value in values:
                probe = socket.socket()
                probes.append(probe)
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", value))
                probe.listen(1)
            return {
                "TopDial": values[:2],
                "DuRecDial2.0": values[2:],
            }
        except OSError:
            pass
        finally:
            for probe in probes:
                probe.close()
    raise OSError("no free EnPL runtime port block")


class Pipeline:
    def __init__(self, root, run_root, python, mode, convplan_zip):
        self.root, self.run_root = Path(root).resolve(), Path(run_root).resolve()
        self.python, self.mode = Path(python).resolve(), mode
        self.convplan_zip = Path(convplan_zip).resolve()
        self.fixed_count = None if mode == "mini" else 400
        self.comparison = self.root / (
            "configs/comparison.enpl-mini.json"
            if mode == "mini"
            else "configs/comparison.enpl.json"
        )
        self.method = self.root / "configs/enpl.tdrl-adapted.json"
        self.source_lock = self.root / "configs/enpl-reference.lock.json"
        self.fixed_setting = self.root / "configs/baseline-fixed1000-test-v1.json"
        self.bank_root = self.run_root / "bank"
        self.bank = self.bank_root / "plan-bank.jsonl"
        self.bank_manifest = self.bank_root / "bank-manifest.json"
        self.logs = self.run_root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.events = self.run_root / "events.jsonl"
        self.status_path = self.run_root / "status.json"
        self.stop_path = self.run_root / "stop.requested"
        self.children = []
        self.started_at = now()
        self.recovery = os.environ.get("ENPL_AUDITED_RESUME") == "1"
        self.recovery_amendment = None

    def event(self, event, **fields):
        value = {
            "timestamp": now(),
            "event": event,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "mode": self.mode,
            **fields,
        }
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(value, sort_keys=True), flush=True)

    def status(self, state, stage, **fields):
        atomic_json(
            self.status_path,
            {
                "schema_version": 1,
                "pipeline": "enpl_reconstruction_fixed400_two_gpu",
                "state": state,
                "stage": stage,
                "mode": self.mode,
                "fixed_test_count": self.fixed_count,
                "started_at": self.started_at,
                "updated_at": now(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "run_root": str(self.run_root),
                **fields,
            },
        )

    def stop(self, signum, _frame):
        self.stop_path.touch(exist_ok=True)
        self.event("stop_requested", signal=signum)
        for child in self.children:
            if child.poll() is None:
                child.terminate()

    def preflight(self):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("EnPL scientific E2E must run inside Slurm")
        dirty = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=self.root,
            text=True,
        ).strip()
        if dirty:
            raise ValueError(
                "EnPL scientific E2E requires a clean committed worktree; "
                "observed entries: " + dirty.replace("\n", "; ")
            )
        self.commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=self.root, text=True
        ).strip()
        required = (
            self.root / "runtime/uv.lock",
            self.root / "toolchains/cuda-12.9.1/bin/nvcc",
            self.root / "configs/models.lock.json",
            self.root / "configs/runtime.json",
            self.root / "data/processed/TopDial/dev.jsonl",
            self.root / "data/processed/DuRecDial2.0/dev.jsonl",
            self.root / "data/processed/DuRecDial2.0/train.jsonl",
            self.convplan_zip,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing EnPL runtime assets: " + ", ".join(missing))
        reference = verify_reference_lock(self.source_lock)
        if file_hash(self.convplan_zip) != reference["files"]["ConvPlan.zip"]:
            raise ValueError("ConvPlan archive differs from the pinned release")
        self.setting = (
            None
            if self.mode == "mini"
            else load_fixed_test_setting(self.root, self.fixed_setting, 400)
        )
        metadata_path = self.run_root / "run-metadata.json"
        expected = {
            "git_commit": self.commit,
            "mode": self.mode,
            "convplan_zip_sha256": file_hash(self.convplan_zip),
        }
        if metadata_path.exists():
            current = json.loads(metadata_path.read_text(encoding="utf-8"))
            immutable = {key: value for key, value in expected.items() if key != "git_commit"}
            if any(current.get(key) != value for key, value in immutable.items()):
                raise ValueError("EnPL run root belongs to another source or mode")
            if current.get("git_commit") != self.commit:
                if not self.recovery or self.mode != "initial400":
                    raise ValueError("EnPL run root belongs to another source or mode")
                selection = self.run_root / "evaluation/method.lock.json"
                envelope = verify_selection(selection)
                if envelope["payload"]["code"]["commit"] != current["git_commit"]:
                    raise ValueError("EnPL recovery selection/source mismatch")
                prior_path = self.run_root / "recovery-amendment.json"
                if not prior_path.is_file():
                    raise ValueError("EnPL chained recovery lost its original amendment")
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
                if (
                    prior.get("schema_version") != 1
                    or prior.get("selection_sha256") != envelope["sha256"]
                    or prior.get("original_code") != envelope["payload"]["code"]
                ):
                    raise ValueError("EnPL original recovery amendment is invalid")
                unresolved = prior.get("unresolved_requests", [])
                if not unresolved or any(
                    not row["record"]["request_id"].startswith("test/EnPL/")
                    for row in unresolved
                ):
                    raise ValueError("EnPL recovery request inventory is invalid")
                expected_pending = {row["path"]: row for row in unresolved}
                actual_pending = {
                    str(path.relative_to(self.run_root)): path
                    for path in (self.run_root / "evaluation/test-workers").rglob(
                        "pending-*.json"
                    )
                }
                if any(path not in expected_pending for path in actual_pending):
                    raise ValueError("EnPL recovery contains a foreign pending request")
                for relative, row in expected_pending.items():
                    pending_path = self.run_root / relative
                    archive_path = (
                        pending_path.parent.parent
                        / "recovery-ledger"
                        / pending_path.name
                    )
                    existing = [
                        path for path in (pending_path, archive_path) if path.is_file()
                    ]
                    if len(existing) != 1 or file_hash(existing[0]) != row["sha256"]:
                        raise ValueError(
                            "EnPL recovery request is neither exact pending nor exact archived"
                        )
                event_rows = [
                    json.loads(line)
                    for line in self.events.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                prior_events = [
                    row for row in event_rows if row.get("job_id") == "60468"
                ]
                if (
                    not any(
                        row.get("event") == "pipeline_failed"
                        and "post-lock-smoke" in row.get("message", "")
                        for row in prior_events
                    )
                    or any(
                        row.get("event") == "stage_started"
                        and str(row.get("stage", "")).startswith("evaluate-test")
                        for row in prior_events
                    )
                ):
                    raise ValueError("EnPL prior recovery execution audit failed")
                self.recovery_amendment = (
                    self.run_root
                    / "recovery-amendments"
                    / f"{self.commit}.json"
                )
                amendment = {
                    "schema_version": 1,
                    "authorization": (
                        "user-requested audited continuation after the idempotent "
                        "smoke replay failure in job 60468"
                    ),
                    "source_job_id": "60429",
                    "prior_recovery_job_id": "60468",
                    "prior_recovery_amendment_sha256": file_hash(prior_path),
                    "prior_recovery_events_sha256": digest(prior_events),
                    "prior_recovery_official_test_requests": 0,
                    "original_code": envelope["payload"]["code"],
                    "recovery_code": {"commit": self.commit, "dirty": False},
                    "selection_sha256": envelope["sha256"],
                    "unresolved_requests": unresolved,
                    "repair_scope": (
                        "reissue exact pending payloads; remove literal think transport "
                        "tags only for the audited EnPL planner request"
                    ),
                    "orchestration_fix": (
                        "restore the frozen per-turn planner seed context before "
                        "direct journal replay of a completed smoke dialogue"
                    ),
                    "prompts_models_data_selection_changed": False,
                }
                if self.recovery_amendment.exists():
                    if json.loads(self.recovery_amendment.read_text()) != amendment:
                        raise ValueError("EnPL recovery amendment changed")
                else:
                    atomic_json(self.recovery_amendment, amendment)
        else:
            atomic_json(
                metadata_path,
                {
                    "schema_version": 1,
                    "pipeline": "EnPL-domain3-independent-reconstruction-adapted",
                    **expected,
                    "created_at": now(),
                    "initial_job_id": os.environ["SLURM_JOB_ID"],
                    "fixed_prefix": self.fixed_count,
                    "diagnostic_only": self.mode == "mini",
                    "training": "none",
                },
            )

    def prepare_bank(self):
        self.status("running", "prepare-bank")
        self.event("stage_started", stage="prepare-bank")
        if self.bank_manifest.exists():
            load_bank(self.bank, self.bank_manifest)
        else:
            subprocess.run(
                [
                    str(self.python),
                    "scripts/enpl/prepare_bank.py",
                    "--root",
                    str(self.root),
                    "--convplan-zip",
                    str(self.convplan_zip),
                    "--output",
                    str(self.bank_root),
                ],
                cwd=self.root,
                check=True,
            )
        self.event("stage_completed", stage="prepare-bank")

    def materialize_runtime(self):
        base = json.loads((self.root / "configs/runtime.json").read_text())
        assigned = ports(os.environ["SLURM_JOB_ID"])
        self.runtime = {}
        for dataset, values in assigned.items():
            config = json.loads(json.dumps(base))
            config["generation"]["port"] = values[0]
            config["embedding"]["port"] = values[1]
            path = self.run_root / "runtime" / os.environ["SLURM_JOB_ID"] / f"{dataset}.json"
            atomic_json(path, config)
            self.runtime[dataset] = path
        atomic_json(
            self.run_root / "runtime" / os.environ["SLURM_JOB_ID"] / "ports.json",
            {"schema_version": 1, "ports": assigned},
        )

    @staticmethod
    def srun(command):
        return [
            "srun", "--exclusive", "--exact", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=8", "--gpus-per-task=1", "--gpu-bind=single:1",
            "--mem=64G", *command,
        ]

    def parallel(self, stage, commands):
        if self.stop_path.exists():
            raise InterruptedError(f"stop requested before {stage}")
        self.status("running", stage, workers=list(commands))
        self.event("stage_started", stage=stage, workers=list(commands))
        handles = []
        self.children = []
        try:
            for name, command in commands.items():
                handle = (self.logs / f"{stage}-{name}-{os.environ['SLURM_JOB_ID']}.log").open(
                    "a", encoding="utf-8"
                )
                handles.append(handle)
                self.children.append(
                    subprocess.Popen(
                        self.srun(command), cwd=self.root, stdout=handle,
                        stderr=subprocess.STDOUT, text=True
                    )
                )
            while True:
                codes = [child.poll() for child in self.children]
                failure = next((code for code in codes if code not in (None, 0)), None)
                if failure is not None:
                    for child in self.children:
                        if child.poll() is None:
                            child.terminate()
                    raise subprocess.CalledProcessError(failure, stage)
                if all(code == 0 for code in codes):
                    break
                if self.stop_path.exists():
                    raise InterruptedError(f"stop requested during {stage}")
                time.sleep(2)
        finally:
            for child in self.children:
                if child.poll() is None:
                    child.terminate()
            for handle in handles:
                handle.close()
            self.children = []
        self.event("stage_completed", stage=stage)

    def common(self):
        return [
            "--root", str(self.root),
            "--comparison-config", str(self.comparison),
            "--enpl-config", str(self.method),
            "--source-lock", str(self.source_lock),
            "--bank", str(self.bank),
            "--bank-manifest", str(self.bank_manifest),
        ]

    def evaluate(self, split, selection=None):
        commands = {}
        for dataset in DATASETS:
            output = self.run_root / "evaluation" / f"{split}-workers" / dataset
            command = [
                str(self.python), "-m", "tdrl.baselines.enpl.evaluate_cli", "evaluate",
                *self.common(), "--output", str(output), "--split", split,
                "--dataset", dataset, "--runtime-config", str(self.runtime[dataset]),
            ]
            if selection:
                command += ["--selection-lock", str(selection)]
            if split == "test":
                command += [
                    "--fixed-test-setting", str(self.fixed_setting),
                    "--fixed-test-count", "400",
                ]
                if self.recovery_amendment and self.recovery_amendment.exists():
                    command += [
                        "--recovery-amendment",
                        str(self.recovery_amendment),
                    ]
            commands[dataset] = command
        prefix = 400 if split == "test" else self.mode
        self.parallel(f"evaluate-{split}-prefix-{prefix}", commands)
        leaf = "test-fixed1000" if split == "test" else "dev"
        name = "enpl-report.json" if split == "dev" else "enpl-report-fixed1000-prefix-400.json"
        inputs = [
            self.run_root / "evaluation" / f"{split}-workers" / dataset / leaf / name
            for dataset in DATASETS
        ]
        output = self.run_root / "evaluation" / leaf / name
        subprocess.run(
            [str(self.python), "scripts/enpl/merge_evaluation.py", "--input", str(inputs[0]),
             "--input", str(inputs[1]), "--output", str(output)],
            cwd=self.root, check=True,
        )
        return output

    def lock(self, development):
        path = self.run_root / "evaluation/method.lock.json"
        if path.exists():
            verify_selection(path)
            return path
        subprocess.run(
            [str(self.python), "-m", "tdrl.baselines.enpl.evaluate_cli", "lock-selection",
             *self.common(), "--development-evidence", str(development), "--output", str(path)],
            cwd=self.root, check=True,
        )
        return path

    def smoke(self, selection):
        commands = {
            dataset: [
                str(self.python), "scripts/enpl/smoke.py", "--root", str(self.root),
                "--output", str(self.run_root / "smoke" / dataset), "--dataset", dataset,
                "--selection-lock", str(selection), "--source-lock", str(self.source_lock),
                "--enpl-config", str(self.method), "--bank", str(self.bank),
                "--bank-manifest", str(self.bank_manifest), "--runtime-config",
                str(self.runtime[dataset]), "--comparison-config", str(self.comparison),
            ]
            for dataset in DATASETS
        }
        self.parallel("post-lock-smoke", commands)

    def feasibility(self, development):
        report = json.loads(Path(development).read_text(encoding="utf-8"))
        config = json.loads(self.comparison.read_text(encoding="utf-8"))
        dev = config["development_scenarios"] * len(config["protocols"])
        projected = {
            dataset: report["parallel_worker_wall_s"][dataset] / dev * 400
            for dataset in DATASETS
        }
        end_time = os.environ.get("SLURM_JOB_END_TIME")
        if not end_time or not end_time.isdigit():
            raise RuntimeError("Slurm did not expose a numeric allocation end time")
        remaining = max(0.0, float(end_time) - time.time())
        required = (max(projected.values()) + 2 * 3600) * 1.25
        payload = {
            "schema_version": 1,
            "status": "passed" if required <= remaining else "failed",
            "projected_worker_seconds": projected,
            "required_wall_seconds": required,
            "remaining_allocation_seconds": remaining,
            "fits": required <= remaining,
        }
        atomic_json(self.run_root / "feasibility-prefix-400.json", payload)
        if not payload["fits"]:
            raise RuntimeError("EnPL fixed-400 test does not fit allocation")

    def run(self):
        self.status("running", "preflight")
        self.preflight()
        self.prepare_bank()
        self.materialize_runtime()
        development = self.run_root / "evaluation/dev/enpl-report.json"
        selection = self.run_root / "evaluation/method.lock.json"
        if not selection.exists():
            development = self.evaluate("dev")
            selection = self.lock(development)
        else:
            envelope = verify_selection(selection)
            if not development.is_file() or file_hash(development) != envelope["payload"][
                "development_evidence_sha256"
            ]:
                raise ValueError("EnPL method lock lost its development evidence")
        self.smoke(selection)
        test_report = None
        if self.mode != "mini":
            self.feasibility(development)
            test_report = self.evaluate("test", selection)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "pipeline": "EnPL-domain3-independent-reconstruction-adapted",
            "mode": self.mode,
            "started_at": self.started_at,
            "ended_at": now(),
            "source_commit": self.commit,
            "bank_manifest": str(self.bank_manifest),
            "bank_sha256": file_hash(self.bank),
            "method_lock": str(selection),
            "development_report": str(development),
            "test_report": str(test_report) if test_report else None,
            "official_test_opened": test_report is not None,
            "claim_limit": "independent reconstruction and adapted shared-protocol result",
        }
        if self.recovery_amendment and self.recovery_amendment.exists():
            summary["recovery_amendment"] = str(self.recovery_amendment)
            summary["recovery_amendment_sha256"] = file_hash(
                self.recovery_amendment
            )
        name = "e2e-summary-mini.json" if self.mode == "mini" else "e2e-summary-fixed400.json"
        atomic_json(self.run_root / name, summary)
        self.status("complete", "complete", summary=str(self.run_root / name))
        self.event("pipeline_completed", summary=str(self.run_root / name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mode", choices=("mini", "initial400"), required=True)
    parser.add_argument("--convplan-zip", type=Path, required=True)
    args = parser.parse_args()
    pipeline = Pipeline(args.root, args.run_root, args.python, args.mode, args.convplan_zip)
    signal.signal(signal.SIGUSR1, pipeline.stop)
    signal.signal(signal.SIGTERM, pipeline.stop)
    try:
        pipeline.run()
    except InterruptedError as error:
        pipeline.event("pipeline_stopped", reason=str(error))
        pipeline.status("partial", "stopped", reason=str(error))
        raise SystemExit(75) from error
    except Exception as error:
        pipeline.event("pipeline_failed", error=type(error).__name__, message=str(error))
        pipeline.status("failed", "failed", error=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
