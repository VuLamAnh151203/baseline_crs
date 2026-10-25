"""Fresh audited protocol restart and real-H100 judge/controller gates."""

import gc
import json
import os
import shutil
import time

import numpy as np
import torch

from .comparison import Comparison
from .contracts import Message
from .evaluation import PPDPPPlanner, initial_state, policy_input, run_dialogue, visible_context
from .judge import LocalJudge
from .policy_optimizer import policy_optimizer
from .policy_training import load_policy, policy_backward
from .ppdpp import reinforce_loss
from .provenance import atomic_json, digest, file_hash, validate_manifest
from .runtime import Servers
from .stages import code_provenance, counters
from .state import DialogueState
from .test_sampling import projected_sample_counts
from .training import load_prepared


def replay_policy(policy, tokenizer, scenario):
    """Two actual BF16 forwards/backwards; no optimizer update and no generation."""
    from .recovery import gradient_fingerprint

    rows = []
    policy._recovery_execution = {"deterministic_backward": True}
    policy.train()
    policy.policy.gradient_checkpointing_enable()
    ids, mask = policy_input(tokenizer, initial_state(scenario), visible_context(scenario))
    for _ in range(2):
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        generator = torch.Generator(device="cuda").manual_seed(29)
        actions, logs = [], []
        for _ in range(3):
            action, log = policy.select(ids, mask, generator)
            actions.append(int(action.item()))
            logs.append(log.squeeze())
        values = [float(x.detach()) for x in logs]
        loss = reinforce_loss(logs, [-1.0, 0.5, 1.0], 0.999)
        policy_backward(policy, loss)
        gradients = {
            n: p.grad.detach().cpu() for n, p in policy.named_parameters() if p.grad is not None
        }
        if not all(torch.isfinite(g).all() for g in gradients.values()):
            raise ValueError("nonfinite native policy smoke gradient")
        rows.append(
            {
                "actions": actions,
                "log_probabilities": values,
                "gradient_sha256": gradient_fingerprint(gradients),
            }
        )
    policy.zero_grad(set_to_none=True)
    if rows[0] != rows[1]:
        raise ValueError("native policy exact action/logprob/gradient replay failed")
    return {"status": "passed", "no_optimizer_updates": True, "replay": rows[0]}


def policy_replay_gate(policy, tokenizer, comparison, dataset, stage):
    record = replay_policy(policy, tokenizer, load_prepared(comparison.root, dataset, "train")[0])
    atomic_json(stage.directory / "policy-replay.json", record)


def judge_smoke(comparison, stage):
    # Reserve all FP32 optimizer state while both BF16 Qwen models are resident.
    policy, tokenizer, _ = load_policy(comparison.root)
    optimizer = policy_optimizer(policy, comparison.ppdpp_cfg(), "rl")
    optimizer.reserve()
    replay = replay_policy(policy, tokenizer, load_prepared(comparison.root, "TopDial", "train")[0])
    scenario = {"target": ["Movie recommendation", "The Star"], "profile": {}, "knowledge": []}
    cases = [
        ("explicit", "I'll watch The Star tonight.", True, False),
        ("pronoun", "I'll definitely give it a watch this weekend.", True, False),
        ("mention", "The Star has an interesting cast.", False, False),
        ("conditional", "Maybe I'll watch The Star if I have time.", False, False),
        ("refusal", "Please stop recommending films. I don't want to watch it.", False, True),
    ]
    records = []
    with Servers(comparison.root, stage.directory / "servers", comparison.settings) as servers:
        b = servers.backends()
        judge = LocalJudge(b["generation"], comparison.cfg["judge"])
        for name, text, accepted, refused in cases:
            state = DialogueState(
                (
                    Message("user", "Suggest a film."),
                    Message("assistant", "Would you like to watch The Star?"),
                    Message("user", text),
                )
            )
            verdict = judge.assess(state, scenario, 17, "judge-smoke/" + name)
            replayed = judge.assess(state, scenario, 17, "judge-smoke/" + name)
            if verdict != replayed or verdict.accepted != accepted or verdict.refused != refused:
                raise ValueError(f"semantic judge smoke/replay failed: {name}: {verdict}")
            if accepted and verdict.label != "acceptance":
                raise ValueError(
                    "recommendation fixture must be acceptance, not generic goal completion"
                )
            records.append({"fixture": name, "judgment": verdict.__dict__})
        feedback = judge.feedback(state, scenario, 17, "judge-smoke/critic")
        if feedback != judge.feedback(state, scenario, 17, "judge-smoke/critic"):
            raise ValueError("sampled critic exact journal replay failed")
        truncated = DialogueState(
            (
                Message("user", "Suggest music."),
                Message("assistant", "You could choose Revolution or You're Not Here."),
                Message("user", "I enjoy both. In fact, I'd like to play"),
            )
        )
        result = judge.assess(
            truncated,
            {
                "target": ["Music recommendation", "You're Not Here"],
                "profile": {},
                "knowledge": [],
                "latest_user_reply_finish_reason": "length",
            },
            17,
            "judge-smoke/truncated-ambiguous",
        )
        if result.accepted:
            raise ValueError("truncated ambiguous commitment must not become success")
        records.append({"fixture": "truncated-ambiguous", "judgment": result.__dict__})
        for name, target, assistant, user in [
            (
                "other-song",
                ["Play music", "Beautiful"],
                "Would you like to play Beautiful?",
                "Skip Beautiful. Let's start Revolution instead.",
            ),
            (
                "wrong-goal",
                ["POI recommendation", "Unforgettable Hunan Restaurant(Powerlong City Plaza)"],
                "Would you like to visit Unforgettable Hunan Restaurant?",
                "I'm going to stay home and play Peking Opera Blues instead.",
            ),
            (
                "already-playing",
                ["Play music", "Beautiful"],
                "Try Beautiful.",
                "I've already listened to Beautiful many times. I'm currently listening to Beautiful.",
            ),
        ]:
            observed = DialogueState(
                (
                    Message("user", "Let's talk."),
                    Message("assistant", assistant),
                    Message("user", user),
                )
            )
            assessment = judge.assess(
                observed,
                {"target": target, "profile": {}, "knowledge": []},
                17,
                "judge-smoke/" + name,
            )
            if assessment.accepted:
                raise ValueError(f"wrong-topic/prior-consumption false success: {name}")
            records.append({"fixture": name, "judgment": assessment.__dict__})
        # Required live embedding remains active alongside judge and controller.
        from .planner import INSTRUCTION

        b["embedding"].embed([state.serialize()], INSTRUCTION, "judge-smoke", "judge-smoke/live")
        compute = {**counters(b["generation"], b["embedding"]), **servers.compute_metadata()}
    atomic_json(
        stage.directory / "checks.json",
        {
            "status": "passed",
            "scientific_result": False,
            "fixtures": records,
            "critic": feedback,
            "controller_replay": replay,
            "optimizer_state_precision": "float32",
            "forward_precision": "bfloat16",
            "controller_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    )
    del optimizer, policy
    gc.collect()
    torch.cuda.empty_cache()
    return compute


def budget_preflight(comparison, stage):
    """Small development proxy; never open test bodies or select by proxy SR."""
    summaries, compute = {}, []
    for dataset in comparison.cfg["datasets"]:
        policy, tokenizer, _ = load_policy(comparison.root)
        rows = sorted(
            load_prepared(comparison.root, dataset, "dev"), key=lambda row: digest([17, row["id"]])
        )[:2]
        with Servers(
            comparison.root, stage.directory / "servers" / dataset, comparison.settings
        ) as servers:
            b = servers.backends()
            results = []
            judge = LocalJudge(b["generation"], comparison.cfg["judge"])
            for protocol in comparison.cfg["protocols"]:
                for scenario in rows:
                    actor = PPDPPPlanner(
                        policy.eval(),
                        tokenizer,
                        b["generation"],
                        b["embedding"],
                        visible_context(scenario),
                        comparison.assistant,
                    )
                    results.append(
                        run_dialogue(
                            scenario,
                            actor,
                            b["generation"],
                            b["embedding"],
                            protocol,
                            17,
                            stage.directory / dataset / protocol / scenario["id"],
                            {
                                "phase": "budget-proxy",
                                "method": "PPDPP",
                                "request_namespace": digest(stage.contract),
                            },
                            comparison.cfg["turn_budget"],
                            comparison.simulator,
                            judge=judge,
                        )
                    )
            summaries[dataset] = {
                "dialogues": len(results),
                "warm_p95_seconds": float(np.quantile([r["wall_s"] for r in results], 0.95)),
                "mean_actual_turns": float(np.mean([r["actual_turns"] for r in results])),
            }
            compute.append(
                {**counters(b["generation"], b["embedding"]), **servers.compute_metadata()}
            )
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    data = json.loads((comparison.root / "data/processed/data-manifest.json").read_text())
    counts = {
        d: comparison.cfg["public_dataset_totals"][d]
        - data["datasets"][d]["original_train_count"]
        - data["datasets"][d]["count"]["dev"]
        for d in summaries
    }
    if any(n <= 0 for n in counts.values()):
        raise ValueError("invalid public-metadata test count projection")
    sample_counts = projected_sample_counts(counts, comparison.cfg.get("test_sampling"))
    cells = len(comparison.cfg["seeds"]) * len(comparison.cfg["protocols"]) * 4
    seconds = sum(sample_counts[d] * summaries[d]["warm_p95_seconds"] * cells for d in counts)
    critic_seconds = (
        sum(
            x["latency_s"]
            for p in (comparison.output / "judge-smoke/servers/request-ledger/generation").glob(
                "completed-*.json"
            )
            if (x := json.loads(p.read_text())).get("role") == "critic"
        )
        / 10
    )
    if critic_seconds <= 0:
        raise ValueError("sampled critic timing evidence missing")
    rl = sum(
        1000
        * (
            summaries[d]["warm_p95_seconds"]
            + 10 * critic_seconds * summaries[d]["mean_actual_turns"]
        )
        for d in summaries
    )
    remaining = comparison.cfg["walltime_seconds"] - (time.time() - comparison.allocation_start)
    forecast = (seconds + rl) * comparison.cfg["feasibility_safety_factor"]
    projection = {
        "schema_version": 1,
        "status": "development_proxy_estimate_not_scientific_result",
        "proxy": "untrained policy; real Qwen/judge/embedding; uniform baseline timing for four method/budget cells",
        "test_count_projection": counts,
        "test_sampling": comparison.cfg.get("test_sampling"),
        "evaluated_count_projection": sample_counts,
        "sample_count_projection_basis": "expected pooled uniform composition; exact IDs/counts after lock",
        "development_timing": summaries,
        "test_unique_cells_per_scenario": cells,
        "projected_test_seconds": seconds,
        "projected_native_rl_seconds": rl,
        "critic_seconds_per_sample": critic_seconds,
        "required_seconds_with_safety": forecast,
        "allocation_remaining_seconds": remaining,
        "fitting_reserve_seconds": comparison.cfg["reserve_seconds"],
        "public_count_sources": comparison.cfg["public_dataset_totals_sources"],
        "test_bodies": "sealed",
        "method_specific_postfit_recheck_required": True,
        "fits_proxy": forecast + comparison.cfg["reserve_seconds"] <= remaining,
    }
    atomic_json(stage.directory / "projection.json", projection)
    if not projection["fits_proxy"]:
        raise ValueError(
            "full scoped comparison exceeds remaining allocation in measured development proxy; explicit scope/budget decision required"
        )
    from .stages import combine_compute

    return combine_compute(compute)


class FidelityRepair(Comparison):
    def __init__(self, root, output, config_path, source):
        resume = output.exists()
        source = source.resolve()
        allocation = json.loads((source / "allocation.json").read_text())
        status = json.loads((source / "status.json").read_text())
        if allocation["job_id"] != "57711" or os.environ.get("SLURM_JOB_ID") != "57711":
            raise ValueError("fidelity repair must use allocation 57711")
        if status.get("error") != "allocation stop requested at a durable checkpoint":
            raise ValueError("prior pipeline did not stop at an audited durable checkpoint")
        if (source / "test").exists() or (source / "selection/selection.lock").exists():
            raise ValueError("this amendment requires a pre-lock, pre-test source")
        super().__init__(root, output, config_path)
        if self.cfg["schema_version"] != 2 or code_provenance(root)["dirty"]:
            raise ValueError("fidelity restart requires clean committed v2 source")
        if resume:
            amendment = output / "fidelity-amendment.json"
            old = json.loads(amendment.read_text())
            if (
                old["code"] != code_provenance(root)
                or old["config_sha256"] != file_hash(config_path)
                or old["source"] != str(source)
            ):
                raise ValueError(
                    "fidelity resume code/config/source mismatch; retain old artifacts"
                )
            if list(output.rglob("pending-*.json")):
                raise ValueError(
                    "fidelity resume has unresolved inference outcomes; audit required"
                )
            if json.loads((output / "allocation.json").read_text()) != allocation:
                raise ValueError("fidelity resume allocation clock changed")
            self.recovery_evidence = {
                "path": str(amendment),
                "sha256": file_hash(amendment),
                "source_job_id": "57711",
            }
            self.extra["fidelity_amendment"] = self.recovery_evidence
            self.diagnostic_done = False
            return
        atomic_json(output / "allocation.json", allocation)
        shutil.copy2(source / "slurm-job.txt", output / "slurm-job.txt")
        # Keep the allocation holder's wall-limit signal effective for this new branch.
        (output / "stop.requested").symlink_to(root / "artifacts/recovery-57711/stop.requested")
        checkpoint_rows = {}
        for dataset in self.cfg["datasets"]:
            from .recovery import verified_file

            old_stage = source / "PPDPP-RL" / dataset
            manifest = json.loads((old_stage / "manifest.json").read_text())
            validate_manifest(manifest)
            for path, sha in manifest["outputs"].items():
                verified_file(old_stage / path, sha)
            pending = list((old_stage / "servers/request-ledger").rglob("pending-*.json"))
            if pending:
                raise ValueError("old stopped RL retains unresolved requests; audit before restart")
            checkpoint = source / "PPDPP-RL" / dataset / "checkpoint.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_rows[dataset] = {
                "next_episode": saved["next_episode"],
                "sha256": file_hash(checkpoint),
                "manifest_sha256": file_hash(old_stage / "manifest.json"),
                "unresolved_requests": 0,
            }
            del saved
        amendment = output / "fidelity-amendment.json"
        atomic_json(
            amendment,
            {
                "schema_version": 1,
                "allocation_job_id": "57711",
                "authorization": "user 2026-10-07: fix PPDPP inconsistencies and use local LLM judge",
                "test_sampling_authorization": (
                    "user 2026-10-08: 2000 total official test scenarios, random seed 42"
                    if self.cfg.get("test_sampling") else None
                ),
                "single_seed_authorization": (
                    "user 2026-10-08: all experiments use one seed 17; sampling seed remains 42"
                    if self.cfg["seeds"] == [17] else None
                ),
                "source": str(source),
                "source_status_sha256": file_hash(source / "status.json"),
                "retained_old_rl": checkpoint_rows,
                "code": code_provenance(root),
                "config_sha256": file_hash(config_path),
                "reused_fitting": [],
                "scope": "fresh embeddings/heads/SFT/RL/development/selection/test/report; censored offline terminals; semantic primary outcome; sampled native-code critic",
                "allocation_clock": allocation,
            },
        )
        self.recovery_evidence = {
            "path": str(amendment),
            "sha256": file_hash(amendment),
            "source_job_id": "57711",
        }
        self.extra["fidelity_amendment"] = self.recovery_evidence
        self.diagnostic_done = False

    def stage(self, name, fn, extra=None):
        if name.startswith("embeddings/"):
            super().stage("budget-preflight", lambda st: budget_preflight(self, st))
        if name == "selection" and not self.diagnostic_done:
            self.diagnostic_done = True
            from .recovery import SelectionContinuation

            super().stage(
                "PPDPP-development", lambda st: SelectionContinuation.ppdpp_development(self, st)
            )
        return super().stage(name, fn, extra)
