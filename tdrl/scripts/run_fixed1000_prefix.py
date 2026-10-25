#!/usr/bin/env python3
"""Persistent owned-allocation queue for a resumable frozen-test prefix."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from run_in_allocation import validate_allocation

from tdrl.confirmation_lock import verify_candidate
from tdrl.difftod_evaluation import verify_selected_policy
from tdrl.provenance import atomic_json, canonical, digest, file_hash, validate_manifest
from tdrl.selection import verify_selection
from tdrl.stages import code_provenance
from tdrl.test_prefix import compare_prefix_records, prefix_membership


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()
    root = Path.cwd()
    membership = prefix_membership(args.count)
    config_path = root / "configs/fixed1000-prefix-tdrl-difftod-v1.json"
    cfg = json.loads(config_path.read_text())
    code = code_provenance(root)
    if code["dirty"]:
        raise ValueError("prefix queue requires committed clean source")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / "driver.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job = (root / "logs/active-allocation-job-id.txt").read_text().strip()
    state = {
        "schema_version": 1,
        "job_id": job,
        "code_commit": code["commit"],
        "requested_prefix_count": args.count,
        "membership": membership,
        "status": "queued",
        "phase": "waiting-for-confirmation",
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "started_unix": time.time(),
        "official_test_opened": False,
    }

    def save():
        atomic_json(output / "queue-status.json", state)

    def owned_allocation():
        validate_allocation(
            subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True),
            os.getuid(),
            gpu_count=2,
        )

    def steps():
        return [
            s
            for s in subprocess.check_output(
                ["squeue", "-h", "--steps", "-j", job, "-o", "%i"], text=True
            ).splitlines()
            if not s.endswith((".batch", ".extern"))
        ]

    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
        OMP_NUM_THREADS="4",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        TDRL_POLICY_DETERMINISM="strict-backward-v1",
        TDRL_CANDIDATE_WORKERS="3",
    )

    def parallel(mode, assignments, dataset):
        owned_allocation()
        if steps():
            raise ValueError("unknown active Slurm steps; prefix queue will not attach")
        state.update(status="running", phase=mode)
        save()
        processes, logs = [], []
        for method, slot in assignments:
            log = (output / f"{mode}-{dataset}-{method}-{slot}.log").open("a")
            logs.append(log)
            command = [
                "srun",
                "--jobid=" + job,
                "--ntasks=1",
                "--cpus-per-task=8",
                "--gres=gpu:nvidia_h100_80gb_hbm3:1",
                "--exclusive",
                "--exact",
                "runtime/.venv/bin/python",
                "scripts/evaluate_fixed1000_prefix.py",
                "--mode",
                mode,
                "--method",
                method,
                "--slot",
                str(slot),
                "--dataset",
                dataset,
                "--count",
                str(args.count),
                "--output",
                str(output),
            ]
            processes.append(
                subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            )
        codes = [p.wait() for p in processes]
        for log in logs:
            log.close()
        if any(codes):
            raise ValueError(f"prefix {mode} failed {codes}; all evidence retained")

    def allocated(commands, phase):
        owned_allocation()
        if steps():
            raise ValueError("unknown active steps before fitting stage")
        state.update(status="running", phase=phase)
        save()
        processes, logs = [], []
        for i, command in enumerate(commands):
            log = (output / f"{phase}-{i}.log").open("a")
            logs.append(log)
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
                        *command,
                    ],
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [p.wait() for p in processes]
        for log in logs:
            log.close()
        if any(codes):
            raise ValueError(f"{phase} failed {codes}; evidence retained")

    try:
        save()
        while True:
            owned_allocation()
            queue = json.loads((root / cfg["confirmation_queue"]).read_text())
            if queue["status"] in {"failed", "partial"}:
                raise ValueError("confirmation incomplete/failed; test stays closed")
            if queue["status"] == "complete" and not steps():
                break
            time.sleep(30)
        candidate_path = root / cfg["candidate_lock"]
        candidate = verify_candidate(candidate_path, root)
        # This commit adds only the prefix runner; frozen planner/backends remain exact.
        differences = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                candidate["code_commit"],
                code["commit"],
                "--",
                "tdrl/src/tdrl",
            ],
            text=True,
        ).splitlines()
        if set(differences) - {
            "tdrl/src/tdrl/test_prefix.py",
            "tdrl/src/tdrl/prefix_selection.py",
            "tdrl/src/tdrl/durecdial_diffusion_data.py",
            "tdrl/src/tdrl/difftod_durecdial_evaluation.py",
        }:
            raise ValueError("frozen inference implementation changed after candidate selection")
        _, chosen, _ = verify_selected_policy(root / cfg["difftod_fitting"])
        if chosen["sha256"] != candidate["difftod_policy_sha256"]:
            raise ValueError("DiffTOD selected checkpoint changed")
        candidate_cfg = json.loads((root / cfg["candidate_config"]).read_text())
        if candidate["package_lock_sha256"] != file_hash(root / "runtime/uv.lock"):
            raise ValueError("packages changed since candidate freeze")
        # Train/dev bootstrap is CPU-only, before any new official test opening.
        pin = json.loads((root / "configs/difftod-policy-model.lock.json").read_text())
        tokenizer = root / "models/difftod/gpt2" / pin["tokenizer"]["revision"] / "tokenizer.json"
        prep = root / "artifacts/difftod-durecdial-data-v1"
        state.update(phase="prepare-difftod-durecdial")
        save()
        subprocess.run(
            [
                "runtime/.venv/bin/python",
                "scripts/prepare_difftod_durecdial.py",
                "--tokenizer",
                str(tokenizer),
                "--output",
                str(prep),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        source_manifest = json.loads((prep / "manifest.json").read_text())
        validation = root / "artifacts/difftod-durecdial-validation-v1"
        validation.mkdir(parents=True, exist_ok=True)
        import shutil

        val_path = validation / "validation.npy"
        if val_path.exists():
            if file_hash(val_path) != source_manifest["outputs"]["dev.npy"]:
                raise ValueError("existing DuRecDial validation data changed")
        else:
            shutil.copyfile(prep / "dev.npy", val_path)
        validation_manifest = {
            "schema_version": 1,
            "status": "complete",
            "scope": "DuRecDial development loss-only selection; no official tests",
            "source_manifest_sha256": file_hash(prep / "manifest.json"),
            "source_dev_sha256": source_manifest["outputs"]["dev.npy"],
            "outputs": {"validation.npy": file_hash(validation / "validation.npy")},
        }
        if (validation / "manifest.json").exists():
            if json.loads((validation / "manifest.json").read_text()) != validation_manifest:
                raise ValueError("DuRecDial validation preparation changed")
        else:
            atomic_json(validation / "manifest.json", validation_manifest)
        preflight = root / cfg["durecdial_preflight"]
        allocated(
            [
                [
                    "scripts/preflight_difftod_durecdial.py",
                    "--slot",
                    str(i),
                    "--output",
                    str(preflight / f"smoke-{i}"),
                ]
                for i in (0, 1)
            ],
            "preflight-difftod-durecdial",
        )
        preflight_uuids = {
            json.loads((preflight / f"smoke-{i}/manifest.json").read_text())["provenance"]["gpu"][
                "uuid"
            ]
            for i in (0, 1)
        }
        if len(preflight_uuids) != 2:
            raise ValueError("distinct DuRecDial preflight GPUs required")
        allocated(
            [
                [
                    "scripts/train_difftod_durecdial.py",
                    "--preflight",
                    str(preflight / "smoke-0"),
                    "--output",
                    str(root / cfg["durecdial_fitting"]),
                ]
            ],
            "fit-difftod-durecdial",
        )
        from tdrl.difftod_durecdial_evaluation import verify_selected_policy as verify_dr_policy

        _, dr_chosen, _ = verify_dr_policy(root / cfg["durecdial_fitting"])
        old_lock = root / cfg["durecdial_source"] / "selection/selection.lock"
        old_sha = json.loads(old_lock.read_text())["sha256"]
        legacy = verify_selection(old_lock, old_sha)
        if legacy["selected_head"] != cfg["durecdial_head"]:
            raise ValueError("frozen DuRecDial selected head mismatch")
        dr_head = (
            root
            / cfg["durecdial_source"]
            / f"learning/DuRecDial2.0/{cfg['durecdial_head']}/checkpoint.pt"
        )
        data_manifest = json.loads((root / "data/processed/data-manifest.json").read_text())
        for dataset in cfg["ready_datasets"]:
            if (
                file_hash(root / f"data/raw/{dataset}/data.zip")
                != data_manifest["datasets"][dataset]["archive_sha256"]
            ):
                raise ValueError("official archive changed before selection")
        refs = {
            "prefix_config": config_path,
            "TopDial_archive": root / "data/raw/TopDial/data.zip",
            "DuRecDial_archive": root / "data/raw/DuRecDial2.0/data.zip",
            "data_manifest": root / "data/processed/data-manifest.json",
            "durecdial_legacy_selection": old_lock,
            "durecdial_head": dr_head,
            "durecdial_bank_index": root
            / cfg["durecdial_source"]
            / "embeddings/DuRecDial2.0/bank/index.json",
            "durecdial_fitting_manifest": root / cfg["durecdial_fitting"] / "manifest.json",
            "durecdial_policy_selection": root / cfg["durecdial_fitting"] / "selection.json",
            "durecdial_fitting_config": root / "configs/difftod-durecdial-fit-v1.json",
            "candidate_lock": candidate_path,
            "package_lock": root / "runtime/uv.lock",
            "models": root / "configs/models.lock.json",
            "head": root / candidate_cfg["head_source"] / "MRNFixed/checkpoint.pt",
            "bank_index": root / candidate_cfg["source"] / "embeddings/TopDial/bank/index.json",
            "diffusion_model_pin": root / "configs/difftod-policy-model.lock.json",
            "diffusion_fit_config": root / "configs/difftod-topdial-fit-v1.json",
            "diffusion_fitting_manifest": root / cfg["difftod_fitting"] / "manifest.json",
            "diffusion_selection": root / cfg["difftod_fitting"] / "selection.json",
        }
        report = root / cfg["confirmation_queue"]
        report = report.parent / "paired-report/report.json"
        if not report.is_file():
            raise ValueError("complete independent confirmation report required")
        refs["confirmation_report"] = report
        # Record confirmation regardless of score; no new configuration is selected here.
        payload = {
            "schema_version": 1,
            "scope": "fixed1000-prefix-test",
            "durecdial_head_sha256": file_hash(dr_head),
            "durecdial_policy_sha256": dr_chosen["sha256"],
            "durecdial_selected_head": legacy["selected_head"],
            "selection_rule": "User-selected v12 Full frozen before independent confirmation; no test tuning",
            "code_commit": code["commit"],
            "candidate_payload_sha256": digest(candidate),
            "methods": cfg["methods"],
            "protocols": cfg["protocols"],
            "seed": 17,
            "full_membership_sha256": membership["full_order_sha256"],
            "ready_datasets": cfg["ready_datasets"],
            "files": {
                k: {"path": str(p.relative_to(root)), "sha256": file_hash(p)}
                for k, p in refs.items()
            },
        }
        lock = output / "selection.lock"
        envelope = {"payload": payload, "sha256": digest(payload)}
        if lock.exists():
            if json.loads(lock.read_text()) != envelope:
                raise ValueError("immutable prefix selection mismatch")
        else:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
            with os.fdopen(fd, "wb") as f:
                f.write(canonical(envelope) + b"\n")
                f.flush()
                os.fsync(f.fileno())
        for dataset in cfg["ready_datasets"]:
            for assignments in [[("TDRL", 0), ("TDRL", 1)], [("DiffTOD", 0), ("DiffTOD", 1)]]:
                parallel("smoke", assignments, dataset)
        measured = {}
        counts = {d: sum(x[0] == d for x in membership["prefix"]) for d in cfg["ready_datasets"]}
        for dataset in cfg["ready_datasets"]:
            measured[dataset] = {
                method: max(
                    json.loads((output / f"smoke-{dataset}-{method}-{i}/checks.json").read_text())[
                        "full_turn_proxy_seconds"
                    ]
                    for i in (0, 1)
                )
                for method in cfg["methods"]
            }
        allocation = json.loads((root / "artifacts/two-gpu-58159/allocation.json").read_text())
        remaining = allocation["walltime_seconds"] - (time.time() - allocation["started_unix"])
        required = (sum(counts[d] * 2 * max(measured[d].values()) for d in counts) + 14400) * 1.25
        atomic_json(
            output / f"projection-{args.count}.json",
            {
                "measured_seconds": measured,
                "required_wall_seconds_with_safety": required,
                "remaining_wall_seconds": remaining,
                "fits": required <= remaining,
                "prefix_count": args.count,
                "dataset_counts": counts,
                "scope": "two parallel one-GPU method workers; original allocation clock",
            },
        )
        if required > remaining:
            raise ValueError("prefix does not fit allocation; explicit budget decision required")
        comparisons = {}
        sources = {}
        for dataset in cfg["ready_datasets"]:
            if not counts[dataset]:
                continue
            state["official_test_opened"] = True
            parallel("evaluate", [("TDRL", 0), ("DiffTOD", 1)], dataset)
            records = {}
            for method in cfg["methods"]:
                directory = output / f"prefix-{args.count}-{dataset}-{method}"
                manifest = json.loads((directory / "manifest.json").read_text())
                validate_manifest(manifest)
                if manifest["status"] != "complete":
                    raise ValueError("partial prefix method")
                if file_hash(directory / "records.json") != manifest["outputs"]["records.json"]:
                    raise ValueError("prefix records checksum mismatch")
                records[method] = json.loads((directory / "records.json").read_text())
                sources[dataset + "/" + method] = {
                    "path": str(directory),
                    "manifest_sha256": file_hash(directory / "manifest.json"),
                }
            expected_ids = [sid for d, sid in membership["prefix"] if d == dataset]
            comparisons[dataset] = compare_prefix_records(
                records["TDRL"], records["DiffTOD"], expected_ids, envelope["sha256"]
            )
        atomic_json(
            output / f"paired-report-{args.count}.json",
            {
                "schema_version": 1,
                "scope": membership["claim"],
                "membership": membership,
                "selection_sha256": envelope["sha256"],
                "comparisons": comparisons,
                "sources": sources,
            },
        )
        state.update(status="complete", phase="prefix-complete", ended_unix=time.time())
        save()
    except BaseException as error:
        state.update(status="failed", error=repr(error), ended_unix=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
