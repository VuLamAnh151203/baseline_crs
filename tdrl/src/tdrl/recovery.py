"""User-authorized recovery of allocation 57166; original evidence stays immutable."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from .comparison import DATASETS, Comparison
from .contracts import Decoding
from .evaluation import PPDPPPlanner, visible_context
from .policy_training import deterministic_backward, replay_policy_actions
from .provenance import atomic_json, file_hash, validate_manifest
from .stages import code_provenance

SOURCE_COMMIT = "9285587a3099fde2eb2562aec23799c054848bd0"
AMENDABLE = {
    "AGENTS.md",
    "tdrl/src/tdrl/backend.py",
    "tdrl/src/tdrl/runtime.py",
    "tdrl/src/tdrl/recovery.py",
    "tdrl/src/tdrl/policy_training.py",
    "tdrl/src/tdrl/comparison.py",
    "tdrl/src/tdrl/cli.py",
    "tdrl/scripts/submit_recovery.sh",
    "tdrl/scripts/run_in_allocation.py",
    "tdrl/scripts/slurm/recovery.sbatch",
}


def verified_file(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f"recovery evidence checksum mismatch: {path}")


def verify_source_changes(root):
    checkout = root.parent
    changes = subprocess.check_output(
        ["git", "diff", "--name-only", SOURCE_COMMIT, "HEAD"], cwd=checkout, text=True
    ).splitlines()
    forbidden = [
        p for p in changes if p not in AMENDABLE and not p.startswith(("tdrl/tests/", "tdrl/docs/"))
    ]
    if forbidden:
        raise ValueError(f"unreviewed scientific source amendment: {forbidden}")
    return subprocess.check_output(["git", "diff", SOURCE_COMMIT, "HEAD"], cwd=checkout, text=True)


def gradient_fingerprint(gradients):
    sha = hashlib.sha256()
    for name, gradient in gradients.items():
        sha.update(name.encode())
        sha.update(gradient.contiguous().view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def gradient_differences(left, right):
    if left.keys() != right.keys():
        raise ValueError("replay gradient parameter membership changed")
    rows = []
    for name in left:
        a, b = left[name], right[name]
        if not torch.equal(a, b):
            delta = (a.float() - b.float()).abs()
            rows.append(
                {
                    "parameter": name,
                    "changed_elements": int((a != b).sum()),
                    "elements": a.numel(),
                    "max_abs": float(delta.max()),
                }
            )
    return rows


def replay_pair(policy, tokenizer, scenario, cfg, rng, stored, strict_backward):
    """Measure exact replay and persist sufficient evidence to diagnose a failure."""
    gradients, hashes, log_probs, forward_rngs = None, [], [], []
    differences = []
    for repeat in range(2):
        torch.set_rng_state(rng["cpu_rng"])
        torch.cuda.set_rng_state_all(rng["cuda_rng"])
        policy.zero_grad(set_to_none=True)
        actor = PPDPPPlanner(
            policy,
            tokenizer,
            None,
            None,
            visible_context(scenario),
            Decoding(**cfg["assistant_decoding"]),
            training=True,
            policy_seed=cfg["seed"] + rng["episode"],
        )
        replay_policy_actions(actor, stored, scenario)
        values = torch.stack(actor.log_probabilities)
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite H100 replay log probabilities")
        forward_rngs.append(
            {
                "cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                "cuda": [
                    hashlib.sha256(x.numpy().tobytes()).hexdigest()
                    for x in torch.cuda.get_rng_state_all()
                ],
                "action_generator": hashlib.sha256(
                    actor.sample_generator.get_state().cpu().numpy().tobytes()
                ).hexdigest(),
            }
        )
        with deterministic_backward(strict_backward):
            values.sum().backward()
        current = {}
        for name, parameter in policy.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError("nonfinite H100 replay gradients")
                current[name] = parameter.grad.detach().cpu().clone()
        hashes.append(gradient_fingerprint(current))
        if repeat == 0:
            gradients = current
        else:
            differences = gradient_differences(gradients, current)
        log_probs.append(values.detach().float().cpu().tolist())
        del actor, values
    return {
        "strict_backward": strict_backward,
        "gradient_sha256": hashes,
        "gradient_differences": differences,
        "log_probabilities": log_probs,
        "forward_rng": forward_rngs,
        "exact": hashes[0] == hashes[1]
        and log_probs[0] == log_probs[1]
        and forward_rngs[0] == forward_rngs[1],
    }


def replay_check(policy, tokenizer, scenario, cfg, output, rng_path):
    """Diagnose default backward, then require strict exact H100 replay before updates."""
    from .backend import require_h100_slurm

    require_h100_slurm()
    if (
        os.environ.get("TDRL_POLICY_DETERMINISM") != "strict-backward-v1"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("audited deterministic backward environment is missing")
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    stored = json.loads((output / "dialogue.json").read_text())
    report_path = output.parent.parent / "h100-replay-audit.json"
    report = {
        "schema_version": 2,
        "status": "partial",
        "job_id": os.environ["SLURM_JOB_ID"],
        "episode": rng["episode"],
        "saved_actions_verified": len(stored["turns_trace"]),
        "policy_attention_implementation": getattr(
            policy.policy.config, "_attn_implementation", None
        ),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "optimizer_updates_during_audit": 0,
        "limitation": "matches stored strategies and repeat replay; pre-failure gradients were not saved",
    }
    atomic_json(report_path, report)
    try:
        baseline = replay_pair(policy, tokenizer, scenario, cfg, rng, stored, False)
        report["default_backward"] = baseline
        report["diagnosis"] = (
            "default_backward_gradient_nondeterminism_with_stable_forward"
            if baseline["gradient_sha256"][0] != baseline["gradient_sha256"][1]
            and baseline["log_probabilities"][0] == baseline["log_probabilities"][1]
            and baseline["forward_rng"][0] == baseline["forward_rng"][1]
            else "default_backward_already_exact"
            if baseline["exact"]
            else "default_forward_or_rng_replay_is_unstable"
        )
        atomic_json(report_path, report)
        fixed = replay_pair(policy, tokenizer, scenario, cfg, rng, stored, True)
        report["strict_backward"] = fixed
        report["forward_unchanged_by_backward_fix"] = (
            baseline["log_probabilities"] == fixed["log_probabilities"]
            and baseline["forward_rng"] == fixed["forward_rng"]
        )
        if not fixed["exact"] or not report["forward_unchanged_by_backward_fix"]:
            raise ValueError("H100 strict action/gradient replay is not exact or forward changed")
        report["status"] = "passed"
        policy._recovery_execution = {
            "code": code_provenance(Path.cwd()),
            "job_id": os.environ["SLURM_JOB_ID"],
            "compatibility_contract": "original training contract; audited execution amendment",
            "deterministic_backward": True,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "replay_audit_sha256": None,
        }
    except BaseException as error:
        report.update(status="failed", error=str(error), exception_type=type(error).__name__)
        raise
    finally:
        policy.zero_grad(set_to_none=True)
        torch.set_rng_state(rng["cpu_rng"])
        torch.cuda.set_rng_state_all(rng["cuda_rng"])
        atomic_json(report_path, report)
    policy._recovery_execution["replay_audit_sha256"] = file_hash(report_path)


class AuditedRecovery(Comparison):
    def __init__(self, root, output, config_path, source, audit_path):
        super().__init__(root, output, config_path)
        self.source = source.resolve()
        self.audit = json.loads(audit_path.read_text())
        if self.audit["schema_version"] != 1 or self.audit["job_id"] != "57166":
            raise ValueError("unsupported audited recovery")
        if self.source == output.resolve() or code_provenance(root)["dirty"]:
            raise ValueError("recovery requires a separate output and clean committed source")
        self.diff = verify_source_changes(root)
        contract = self.audit["source_contract"]
        expected = {
            **self.extra,
            "config_sha256": file_hash(config_path),
            "package_lock_sha256": file_hash(root / "runtime/uv.lock"),
            "dataset_archives": {
                d: file_hash(root / "data/raw" / d / "data.zip") for d in DATASETS
            },
        }
        if (
            contract["code"] != {"commit": SOURCE_COMMIT, "dirty": False}
            or {k: v for k, v in contract.items() if k != "code"} != expected
        ):
            raise ValueError("recovery model/data/config/serialization/package contracts changed")
        prior = self.audit.get("prior_failed_recovery")
        if not prior or prior["job_id"] != "57561" or prior["new_updates"] != 0:
            raise ValueError("replay fix requires audited failed allocation 57561")
        prior_root = root / prior["directory"]
        for path, sha in prior["files"].items():
            verified_file(prior_root / path, sha)
        previous_stage = prior_root / "PPDPP-RL/DuRecDial2.0"
        verified_file(
            previous_stage / "checkpoint.pt", contract_sha := self.audit["checkpoint"]["sha256"]
        )
        if prior["checkpoint_sha256"] != contract_sha or list(
            (previous_stage / "servers/request-ledger").rglob("*.json")
        ):
            raise ValueError("previous recovery mutated training state or made inference requests")
        previous_manifest = json.loads((previous_stage / "manifest.json").read_text())
        validate_manifest(previous_manifest)
        if (
            previous_manifest["status"] != "failed"
            or previous_manifest.get("error") != "H100 action/gradient replay is not exact"
        ):
            raise ValueError("previous failure is not the authorized replay fix")
        for path, sha in previous_manifest["outputs"].items():
            verified_file(previous_stage / path, sha)
        self.rl_training_contract = contract
        self.rl_recovery_check = replay_check
        self.imports = {}
        self.verify_inputs()

    def verify_inputs(self):
        relative = Path("PPDPP-RL/DuRecDial2.0")
        source = self.source / relative
        if json.loads((source / "contract.json").read_text()) != self.rl_training_contract:
            raise ValueError("source training contract changed")
        verified_file(source / "checkpoint.pt", self.audit["checkpoint"]["sha256"])
        ep = source / "episodes/000243"
        verified_file(ep / "dialogue.json", self.audit["partial_episode"]["dialogue_sha256"])
        verified_file(
            ep / "pre-episode-rng.pt", self.audit["partial_episode"]["pre_episode_rng_sha256"]
        )
        failed = json.loads((source / "manifest.json").read_text())
        validate_manifest(failed)
        if failed["status"] != "failed":
            raise ValueError("source stage failure evidence changed")
        for path, sha in failed["outputs"].items():
            verified_file(source / path, sha)
        pending = sorted((source / "servers/request-ledger").rglob("pending-*.json"))
        records = self.audit["unresolved_requests"]
        if len(pending) != 1 or len(records) != 1:
            raise ValueError("unexpected unresolved inference attempts")
        verified_file(pending[0], records[0]["sha256"])
        if json.loads(pending[0].read_text()) != records[0]["record"]:
            raise ValueError("unresolved request changed")
        saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
        if saved["next_episode"] != 243 or saved["provenance"] != self.rl_training_contract:
            raise ValueError("checkpoint no longer contains exactly 243 committed updates")
        del saved
        self.pending_record = records[0]["record"]
        names = [f"embeddings/{d}" for d in DATASETS]
        names += [f"learning/{d}/{h}" for d in DATASETS for h in ("MRNFixed", "IQE")]
        names += [f"PPDPP-SFT/{d}" for d in DATASETS] + ["PPDPP-RL/TopDial"]
        for name in names:
            directory = self.source / name
            m = json.loads((directory / "manifest.json").read_text())
            validate_manifest(m)
            if m["status"] != "complete":
                raise ValueError(f"recovery import is incomplete: {name}")
            c = json.loads((directory / "contract.json").read_text())
            if c != self.rl_training_contract:
                raise ValueError(f"recovery import contract mismatch: {name}")
            for path, sha in m["outputs"].items():
                verified_file(directory / path, sha)
            target = self.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != directory:
                    raise ValueError("recovery import destination already exists")
            else:
                target.symlink_to(directory, target_is_directory=True)
            self.imports[name] = {
                "source": str(directory),
                "run_id": m["run_id"],
                "manifest_sha256": file_hash(directory / "manifest.json"),
            }
        destination = self.output / relative
        if destination.exists():
            raise ValueError(
                "recovery output already used; retain it and audit before resubmission"
            )
        destination.mkdir(parents=True)
        for name in ("checkpoint.pt", "episodes", "progress.json"):
            src, dst = source / name, destination / name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        journal = source / "servers/generation-journal"
        if journal.exists():
            shutil.copytree(journal, destination / "servers/generation-journal")
        atomic_json(
            self.output / "recovery-amendment.json",
            {
                "schema_version": 1,
                "authorization": "user 2026-10-05: recovery co audit de chay 13 episodes con lai",
                "source_job_id": "57166",
                "failed_recovery_job_id": "57561",
                "numerical_execution_amendment": "controller-only CUBLAS workspace :4096:8 and strict deterministic backward for remaining PPDPP updates; exact replay and unchanged forward required",
                "recovery_job_id": os.environ.get("SLURM_JOB_ID"),
                "source": str(self.source),
                "source_audit": self.audit,
                "code": code_provenance(self.root),
                "imports": self.imports,
                "unknown_failed_request_outcome": self.pending_record,
                "compute_scope": "new stage counters cover recovery segment only; source ledger retained separately; one prior embedding attempt has unknown outcome",
                "protocol_amendment": "spans allocations; transport and recovery instrumentation changed; scientific model/data/prompts/config unchanged",
            },
        )
        (self.output / "source-amendment.diff").write_text(self.diff)
        self.recovery_evidence = {
            "path": str(self.output / "recovery-amendment.json"),
            "sha256": file_hash(self.output / "recovery-amendment.json"),
            "source_job_id": "57166",
            "source_diff_sha256": file_hash(self.output / "source-amendment.diff"),
        }

    def stage(self, name, fn, extra=None):
        if name in self.imports:
            return SimpleNamespace(directory=self.output / name, reused=True)
        return super().stage(
            name, fn, {**(extra or {}), "recovery_amendment": self.recovery_evidence}
        )

    def configure_rl_backends(self, backends):
        backends["embedding"].authorized_reissue = self.pending_record

    def report(self, stage):
        result = super().report(stage)
        atomic_json(stage.directory / "recovery-context.json", self.recovery_evidence)
        return result

    def rl(self, stage, dataset):
        if dataset != "DuRecDial2.0":
            raise ValueError("only DuRecDial recovery is authorized")
        result = super().rl(stage, dataset)
        audit = stage.directory / "h100-replay-audit.json"
        if (
            not audit.exists()
            or not (stage.directory / "servers/request-ledger/authorized-reissue.json").exists()
        ):
            raise ValueError("H100 recovery evidence incomplete")
        saved = torch.load(
            stage.directory / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        if saved["next_episode"] != 256:
            raise ValueError("13 remaining RL episodes were not completed")
        atomic_json(
            stage.directory / "recovery-completion.json",
            {
                "status": "complete",
                "preserved_updates": 243,
                "additional_updates": 13,
                "next_episode": 256,
                "source_checkpoint_sha256": self.audit["checkpoint"]["sha256"],
                "recovered_checkpoint_sha256": file_hash(stage.directory / "checkpoint.pt"),
                "compute_scope": "recovery segment; prior unknown attempt retained in recovery-amendment.json",
            },
        )
        return result


class SelectionContinuation(Comparison):
    """Invalidate failed selection; retain hash-verified completed fitting in allocation 57711."""

    def __init__(self, root, output, config_path, source, repair=False):
        if output.exists():
            raise ValueError(
                "selection-fix destination already exists; audit before another attempt"
            )
        super().__init__(root, output, config_path)
        self.source = source.resolve()
        old_allocation = json.loads((self.source / "allocation.json").read_text())
        if old_allocation["job_id"] != "57711" or os.environ.get("SLURM_JOB_ID") != "57711":
            raise ValueError("selection continuation must stay in allocation 57711")
        failed_path = self.source / "selection/manifest.json"
        failed = json.loads(failed_path.read_text())
        validate_manifest(failed)
        if failed["status"] != "failed" or failed.get("error") != (
            "zero development successes: compute-per-success tie-breaker unresolved"
            if repair
            else "request ID reused with different inputs"
        ):
            raise ValueError("selection failure is not the authorized namespace amendment")
        if (self.source / "selection/selection.lock").exists() or (self.source / "test").exists():
            raise ValueError("selection amendment is restricted to pre-lock, pre-test failure")
        changes = subprocess.check_output(
            ["git", "diff", "--name-only", "8a1e31a" if repair else "ceba35f", "HEAD"],
            cwd=root.parent,
            text=True,
        ).splitlines()
        allowed = {
            "AGENTS.md",
            "tdrl/src/tdrl/cli.py",
            "tdrl/src/tdrl/comparison.py",
            "tdrl/src/tdrl/evaluation.py",
            "tdrl/src/tdrl/runtime.py",
            "tdrl/src/tdrl/recovery.py",
        }
        if repair:
            allowed |= {
                "tdrl/src/tdrl/graph.py",
                "tdrl/src/tdrl/reachability.py",
                "tdrl/src/tdrl/selection.py",
                "tdrl/src/tdrl/training.py",
                "tdrl/src/tdrl/planner.py",
                "tdrl/src/tdrl/autonomy.py",
            }
        if any(
            p not in allowed and not p.startswith(("tdrl/tests/", "tdrl/docs/")) for p in changes
        ):
            raise ValueError("unreviewed source change for selection continuation")
        if code_provenance(root)["dirty"]:
            raise ValueError("selection continuation requires committed clean source")
        for path, sha in failed["outputs"].items():
            verified_file(self.source / "selection" / path, sha)
        expected = {
            **self.extra,
            "config_sha256": file_hash(config_path),
            "package_lock_sha256": file_hash(root / "runtime/uv.lock"),
            "dataset_archives": {
                d: file_hash(root / "data/raw" / d / "data.zip") for d in DATASETS
            },
        }
        prior_amendment = self.source / (
            "selection-amendment.json" if repair else "recovery-amendment.json"
        )
        self.imports = {}
        names = [f"embeddings/{d}" for d in DATASETS]
        names += [f"learning/{d}/{h}" for d in DATASETS for h in ("MRNFixed", "IQE")]
        names += [f"PPDPP-SFT/{d}" for d in DATASETS]
        if not repair:
            names += [f"PPDPP-RL/{d}" for d in DATASETS]
        for name in names:
            directory = (self.source / name).resolve()
            m = json.loads((directory / "manifest.json").read_text())
            validate_manifest(m)
            if m["status"] != "complete":
                raise ValueError(f"incomplete fitting artifact: {name}")
            c = json.loads((directory / "contract.json").read_text())
            if {k: v for k, v in c.items() if k not in {"code", "recovery_amendment"}} != expected:
                raise ValueError(f"fitting scientific contract changed: {name}")
            if c["code"]["dirty"] or c["code"]["commit"] not in {
                SOURCE_COMMIT,
                "ceba35f6c38929459d52ddc1ab8fb83bea40eb21",
            }:
                raise ValueError("unexpected imported source provenance")
            if "recovery_amendment" in c:
                if c["recovery_amendment"]["sha256"] != file_hash(prior_amendment):
                    raise ValueError("prior training recovery amendment changed")
            for path, sha in m["outputs"].items():
                verified_file(directory / path, sha)
            target = self.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(directory, target_is_directory=True)
            self.imports[name] = {
                "source": str(directory),
                "run_id": m["run_id"],
                "manifest_sha256": file_hash(directory / "manifest.json"),
            }
        if not repair:
            completed = json.loads(
                (self.source / "PPDPP-RL/DuRecDial2.0/recovery-completion.json").read_text()
            )
            if completed["status"] != "complete" or completed["next_episode"] != 256:
                raise ValueError("DuRecDial RL recovery is incomplete")
            verified_file(
                self.source / "PPDPP-RL/DuRecDial2.0/checkpoint.pt",
                completed["recovered_checkpoint_sha256"],
            )
        # Preserve the original allocation clock: idle/debug time also consumes the 20-day budget.
        atomic_json(self.output / "allocation.json", old_allocation)
        shutil.copy2(self.source / "slurm-job.txt", self.output / "slurm-job.txt")
        (self.output / "stop.requested").symlink_to(self.source / "stop.requested")
        atomic_json(
            self.output / "selection-amendment.json",
            {
                "schema_version": 1,
                "authorization": "user 2026-10-07: debug low SR and simplify MRN/IQE"
                if repair
                else "user 2026-10-06: fix selection request namespace",
                "allocation_job_id": "57711",
                "code": code_provenance(root),
                "invalidated_selection": str(self.source / "selection"),
                "failed_manifest_sha256": file_hash(failed_path),
                "prior_recovery_amendment_sha256": file_hash(prior_amendment),
                "imports": self.imports,
                "scope": "fresh PPDPP RL on both datasets and fresh selection/test/report; prompt and guidance amendment; acceptance v1/models/seeds/config frozen; embedding/head/SFT contracts retained"
                if repair
                else "fresh selection/test/report; fitting retained; decoding seeds/models/prompts/config unchanged",
                "cache_policy": "no old selection cache, dialogue or metrics imported",
            },
        )
        self.recovery_evidence = {
            "path": str(self.output / "selection-amendment.json"),
            "sha256": file_hash(self.output / "selection-amendment.json"),
            "source_job_id": "57711",
        }

        self._compatibility_checked = not repair
        self._baseline_diagnostic_done = not repair

    def check_head_compatibility(self, stage):
        """Compare cached geometry with original formulas on frozen real training embeddings."""
        import math

        from .reachability import assert_invariants, interval_union
        from .training import load_embeddings, load_head

        rows = {}
        for dataset in DATASETS:
            records, vectors = load_embeddings(self.output / "embeddings" / dataset / "bank")
            from .acceptance import classify_user
            from .contracts import Message
            from .state import DialogueState

            for record in records:
                if record["terminal"]:
                    raw = json.loads(record["state"])
                    observed = DialogueState(tuple(Message(**m) for m in raw["history"]))
                    failed = (
                        observed.history[-1].role != "user"
                        or not classify_user(observed, record["target"]).accepted
                    )
                    if failed != record["failed_terminal"]:
                        raise ValueError(
                            "refusal repair changed offline terminal labels; cached fitting cannot be reused"
                        )
            states = vectors[torch.linspace(0, len(vectors) - 1, 32).long()].to(
                "cuda", torch.bfloat16
            )
            for kind in ("MRNFixed", "IQE"):
                head = load_head(self.output / "learning" / dataset / kind / "checkpoint.pt")
                with torch.no_grad():
                    source, target = states[:, None], states[None, :]
                    if kind == "MRNFixed":
                        sym = head.sym(source) - head.sym(target)
                        legacy = torch.linalg.vector_norm(sym, dim=-1) / math.sqrt(sym.shape[-1])
                        legacy += (head.asym(source) - head.asym(target)).clamp_min(0).amax(-1)
                    else:
                        a = head.encoder(source).unflatten(
                            -1, (head.components, head.component_dim)
                        )
                        b = head.encoder(target).unflatten(
                            -1, (head.components, head.component_dim)
                        )
                        components = interval_union(a, b)
                        alpha = head.mix.sigmoid()
                        legacy = alpha * components.amax(-1) + (1 - alpha) * components.mean(-1)
                    encoded = head.encode(states)
                    cached = head.distance_encoded(encoded[:, None], encoded[None, :])
                    if not torch.allclose(legacy.float(), cached.float(), atol=2e-3, rtol=2e-2):
                        raise ValueError("cached quasimetric geometry differs from original head")
                    invariants = assert_invariants(head, states, atol=2e-3, rtol=2e-2)
                rows[dataset + "/" + kind] = {
                    "checkpoint_sha256": head.checkpoint_hash,
                    "max_absolute_difference": float((legacy.float() - cached.float()).abs().max()),
                    "invariants": invariants,
                    "precision": "bfloat16",
                    "states": 32,
                }
                del head, encoded, cached, legacy
            del vectors, states
            torch.cuda.empty_cache()
        atomic_json(
            stage.directory / "checks.json",
            {"status": "passed", "scientific_result": False, "original_formulas": rows},
        )

    def ppdpp_development(self, stage):
        from .metrics import aggregate
        from .policy_training import load_policy
        from .provenance import digest
        from .runtime import Servers
        from .stages import combine_compute, counters
        from .training import load_prepared

        compute = []
        summary = {}
        for dataset in DATASETS:
            checkpoint = self.output / "PPDPP-RL" / dataset / "checkpoint.pt"
            policy, tokenizer, _ = load_policy(self.root, checkpoint)
            rows = sorted(
                load_prepared(self.root, dataset, "dev"), key=lambda s: digest([17, s["id"]])
            )[: self.cfg["development_scenarios"]]
            with Servers(
                self.root, stage.directory / "servers" / dataset, self.settings
            ) as servers:
                backends = servers.backends()
                for protocol in self.cfg["protocols"]:
                    results = []
                    for scenario in rows:
                        results.append(
                            self.evaluate(
                                backends,
                                scenario,
                                "PPDPP",
                                protocol,
                                17,
                                stage.directory / dataset / protocol / scenario["id"],
                                {
                                    "phase": "development-baseline",
                                    "policy_sha256": file_hash(checkpoint),
                                    "budget": "natural",
                                },
                                policy=policy,
                                tokenizer=tokenizer,
                            )
                        )
                    summary[dataset + "/" + protocol] = aggregate(results)
                record = {
                    **counters(backends["generation"], backends["embedding"]),
                    **servers.compute_metadata(),
                }
                compute.append(record)
            del policy
            torch.cuda.empty_cache()
        atomic_json(
            stage.directory / "summary.json",
            {"scope": "development diagnostic; not official test", "results": summary},
        )
        return combine_compute(compute)

    def stage(self, name, fn, extra=None):
        if name == "selection" and not self._baseline_diagnostic_done:
            self._baseline_diagnostic_done = True
            self.stage("PPDPP-development", self.ppdpp_development)
        if name in self.imports and not self._compatibility_checked:
            self._compatibility_checked = True
            self.stage("head-compatibility", self.check_head_compatibility)
        if name in self.imports:
            return SimpleNamespace(directory=self.output / name, reused=True)
        return super().stage(
            name, fn, {**(extra or {}), "selection_amendment": self.recovery_evidence}
        )

    def report(self, stage):
        result = super().report(stage)
        atomic_json(stage.directory / "selection-recovery-context.json", self.recovery_evidence)
        return result
