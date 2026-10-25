#!/usr/bin/env python3
"""Run a bounded, non-scientific DPDP end-to-end integration on train/dev data."""

import argparse
import gc
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from tdrl.autonomy import explicit_refusal
from tdrl.backend import require_h100_slurm
from tdrl.baselines.dpdp.assets import verify_assets
from tdrl.baselines.dpdp.cli import selfplay, train
from tdrl.baselines.dpdp.config import DPDPConfig
from tdrl.baselines.dpdp.dataset import build_transitions, prepare_transitions
from tdrl.baselines.dpdp.planner import DPDPPlanner
from tdrl.baselines.dpdp.rewards import annotate_rewards
from tdrl.baselines.dpdp.runtime import load_inference
from tdrl.comparison_dpdp import validate_config as validate_comparison_config
from tdrl.contracts import Decoding
from tdrl.evaluation import initial_state, run_dialogue
from tdrl.judge import LocalJudge
from tdrl.provenance import atomic_json, digest, file_hash
from tdrl.runtime import Servers

DATASETS = {
    "TopDial": "topdial-mini-e2e.json",
    "DuRecDial2.0": "durecdial-mini-e2e.json",
}
PROTOCOLS = ("profile", "resistant")
BUDGETS = ("natural", "matched")
CURRENT_STAGE = "startup"


def _event(stage, **values):
    global CURRENT_STAGE
    CURRENT_STAGE = stage
    print(json.dumps({"mini_e2e_stage": stage, **values}, sort_keys=True), flush=True)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _select_record(source, dataset, split):
    source_sha256 = file_hash(source)
    rows = sorted(_read_jsonl(source), key=lambda row: digest([dataset, split, row["id"]]))
    for row in rows:
        if build_transitions(row, dataset, split, source_sha256) and not explicit_refusal(
            initial_state(row)
        ):
            return row
    raise ValueError(f"no eligible mini DPDP record in {source}")


def _write_jsonl(path, rows):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite mini input: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    pending.replace(path)


def _materialize_configs(root, output, datasets=DATASETS):
    paths = {}
    for dataset, filename in datasets.items():
        base_path = root / "configs/dpdp" / filename
        natural = DPDPConfig.from_path(base_path)
        if natural.dataset != dataset:
            raise ValueError("mini DPDP config/dataset mismatch")
        matched = natural.with_budget(2, 1)
        if natural.training_sha256 != matched.training_sha256:
            raise ValueError("mini matched budget changed fitted DPDP weights")
        paths[dataset] = {}
        for budget, config in (("natural", natural), ("matched", matched)):
            path = output / "configs" / dataset / f"{budget}.json"
            raw = asdict(config)
            raw["actions"] = list(raw["actions"])
            atomic_json(path, raw)
            paths[dataset][budget] = path
    return paths


def _prepare_inputs(root, output, datasets=DATASETS):
    sources, prepared, selection = {}, {}, {}
    for dataset in datasets:
        sources[dataset], prepared[dataset], selection[dataset] = {}, {}, {}
        for split in ("train", "dev"):
            canonical = root / "data/processed" / dataset / f"{split}.jsonl"
            row = _select_record(canonical, dataset, split)
            source = output / "mini-input" / dataset / f"{split}.jsonl"
            transition_path = output / "prepared" / dataset / f"{split}.jsonl"
            _write_jsonl(source, [row])
            manifest = prepare_transitions(source, transition_path, dataset, split)
            sources[dataset][split] = source
            prepared[dataset][split] = transition_path
            selection[dataset][split] = {
                "scenario_id": row["id"],
                "canonical_source_sha256": file_hash(canonical),
                "mini_source_sha256": file_hash(source),
                "transition_count": manifest["transition_count"],
            }
    atomic_json(
        output / "mini-input/manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "scope": "mini_non_scientific_train_dev_only",
            "selection": selection,
            "official_test_opened": False,
        },
    )
    return sources, prepared, selection


def _annotate(root, output, runtime, configs, prepared, datasets=DATASETS):
    annotated, reports, server_compute = {}, {}, {}
    for dataset in datasets:
        annotated[dataset], reports[dataset] = {}, {}
        config = DPDPConfig.from_path(configs[dataset]["natural"])
        server_dir = output / "reward-annotation" / dataset / "servers"
        with Servers(root, server_dir, runtime, kinds=("generation",)) as servers:
            backend = servers.backends()["generation"]
            for split in ("train", "dev"):
                destination = output / "reward-annotation" / dataset / f"{split}.annotated.jsonl"
                reports[dataset][split] = annotate_rewards(
                    prepared[dataset][split],
                    destination,
                    backend,
                    config.critic_samples,
                    config.gamma,
                    config.offline_seed,
                    output / "reward-annotation" / dataset / f"{split}-critic-journal",
                )
                annotated[dataset][split] = destination
            server_compute[dataset] = servers.compute_metadata()
    return annotated, reports, server_compute


def _cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def _fit(output, model, model_lock, configs, annotated, datasets=DATASETS):
    checkpoints, reports = {}, {}
    for dataset in datasets:
        destination = output / "offline" / dataset
        reports[dataset] = train(
            SimpleNamespace(
                config=configs[dataset]["natural"],
                encoder=model,
                tokenizer=model,
                model_lock=model_lock,
                output=destination,
                train=annotated[dataset]["train"],
                dev=annotated[dataset]["dev"],
                resume=None,
            )
        )
        if reports[dataset]["status"] != "complete":
            raise RuntimeError(f"mini offline training incomplete for {dataset}")
        checkpoints[dataset] = destination / "checkpoint-final.pt"
        _cleanup_cuda()
    return checkpoints, reports


def _selfplay(
    root, output, runtime_path, model, model_lock, configs, sources, offline, datasets=DATASETS
):
    checkpoints, reports = {}, {}
    for dataset in datasets:
        destination = output / "selfplay" / dataset
        reports[dataset] = selfplay(
            SimpleNamespace(
                root=root,
                runtime_config=runtime_path,
                config=configs[dataset]["natural"],
                scenarios=sources[dataset]["train"],
                offline_checkpoint=offline[dataset],
                encoder=model,
                tokenizer=model,
                model_lock=model_lock,
                output=destination,
                resume=None,
            )
        )
        if reports[dataset]["status"] != "complete":
            raise RuntimeError(f"mini self-play incomplete for {dataset}")
        checkpoints[dataset] = destination / "checkpoint-final.pt"
        _cleanup_cuda()
    return checkpoints, reports


def _evaluate(
    root,
    output,
    runtime,
    comparison,
    model,
    model_lock,
    configs,
    sources,
    checkpoints,
    datasets=DATASETS,
):
    rows = []
    destination_root = output / "evaluation"
    with Servers(root, destination_root / "servers", runtime) as servers:
        backends = servers.backends()
        for dataset in datasets:
            policy, tokenizer, _, _, _ = load_inference(
                configs[dataset]["natural"],
                checkpoints[dataset],
                model,
                model,
                model_lock,
                "cuda",
            )
            policy.to(dtype=torch.bfloat16).eval()
            scenario = _read_jsonl(sources[dataset]["dev"])[0]
            parsed = {
                budget: DPDPConfig.from_path(path) for budget, path in configs[dataset].items()
            }
            checkpoint_sha256 = file_hash(checkpoints[dataset])
            for protocol in PROTOCOLS:
                for budget in BUDGETS:
                    config = parsed[budget]
                    namespace = digest(
                        {
                            "scope": "mini_non_scientific",
                            "dataset": dataset,
                            "protocol": protocol,
                            "budget": budget,
                            "checkpoint_sha256": checkpoint_sha256,
                            "dpdp_config_sha256": config.sha256,
                        }
                    )
                    planner = DPDPPlanner(
                        backends["generation"],
                        policy,
                        tokenizer,
                        config,
                        scenario,
                        checkpoint_sha256,
                        Decoding(**comparison["assistant_decoding"]),
                        protocol,
                    )
                    destination = destination_root / dataset / protocol / budget
                    result = run_dialogue(
                        scenario,
                        planner,
                        backends["generation"],
                        backends["embedding"],
                        protocol,
                        17,
                        destination,
                        {
                            "phase": "mini_non_scientific",
                            "method": "DPDP",
                            "selection_sha256": "not_applicable",
                            "budget": budget,
                            "target_nodes": [],
                            "request_namespace": namespace,
                        },
                        turn_budget=1,
                        simulator_decoding=Decoding(**comparison["simulator_decoding"]),
                        judge=LocalJudge(backends["generation"], comparison["judge"]),
                    )
                    selected = [
                        turn["plan"]["compute"].get("selected_system")
                        for turn in result["turns_trace"]
                    ]
                    if result["status"] != "complete" or "system2" not in selected:
                        raise RuntimeError(
                            f"mini evaluation did not complete System-2 for {dataset}/{protocol}/{budget}"
                        )
                    rows.append(
                        {
                            "dataset": dataset,
                            "protocol": protocol,
                            "budget": budget,
                            "scenario_id": scenario["id"],
                            "status": result["status"],
                            "accepted": result["accepted"],
                            "outcome_label": result["outcome_label"],
                            "actual_turns": result["actual_turns"],
                            "generation_requests": result["generation_requests"],
                            "total_tokens": result["total_tokens"],
                            "selected_systems": selected,
                            "dialogue_sha256": file_hash(destination / "dialogue.json"),
                        }
                    )
            del policy
            _cleanup_cuda()
        server_compute = servers.compute_metadata()
    expected = {
        (dataset, protocol, budget)
        for dataset in datasets
        for protocol in PROTOCOLS
        for budget in BUDGETS
    }
    observed = {(row["dataset"], row["protocol"], row["budget"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("mini evaluation matrix is incomplete")
    report = {
        "schema_version": 1,
        "status": "mini_non_scientific",
        "official_test_opened": False,
        "turn_budget": 1,
        "dialogues": len(rows),
        "matrix": rows,
        "server_compute": server_compute,
    }
    atomic_json(destination_root / "mini-evaluation-report.json", report)
    return report


def run(args):
    root = args.root.resolve()
    output = args.output.resolve()
    datasets = (
        {args.dataset: DATASETS[args.dataset]}
        if args.dataset is not None
        else DATASETS
    )
    if any(
        (output / name).exists()
        for name in ("mini-e2e-summary.json", "mini-e2e-failure.json", "mini-input", "configs")
    ):
        raise FileExistsError("mini E2E output already contains a run")
    output.mkdir(parents=True, exist_ok=True)
    gpu = require_h100_slurm()
    runtime_path = (
        args.runtime_config.resolve()
        if args.runtime_config is not None
        else root / "configs/runtime.json"
    )
    model_lock = root / "configs/dpdp.models.lock.json"
    model = root / "models/roberta-large" / args.roberta_revision
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    comparison = json.loads((root / "configs/comparison.dpdp.json").read_text(encoding="utf-8"))
    verify_assets(model_lock, model, model)
    validate_comparison_config(comparison)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    _event("prepare", status="started")
    configs = _materialize_configs(root, output, datasets)
    sources, prepared, selected = _prepare_inputs(root, output, datasets)
    _event("prepare", status="complete")

    gpu_work_started_at = datetime.now(timezone.utc)
    _event("reward_annotation", status="started")
    annotated, annotation_reports, annotation_compute = _annotate(
        root, output, runtime, configs, prepared, datasets
    )
    _event("reward_annotation", status="complete")

    _event("offline_training", status="started")
    offline, offline_reports = _fit(
        output, model, model_lock, configs, annotated, datasets
    )
    _event("offline_training", status="complete")

    _event("selfplay", status="started")
    checkpoints, selfplay_reports = _selfplay(
        root,
        output,
        runtime_path,
        model,
        model_lock,
        configs,
        sources,
        offline,
        datasets,
    )
    _event("selfplay", status="complete")

    _event("evaluation", status="started")
    evaluation = _evaluate(
        root,
        output,
        runtime,
        comparison,
        model,
        model_lock,
        configs,
        sources,
        checkpoints,
        datasets,
    )
    _event("evaluation", status="complete", dialogues=evaluation["dialogues"])
    gpu_work_ended_at = datetime.now(timezone.utc)

    summary = {
        "schema_version": 1,
        "status": "mini_non_scientific",
        "scope": {
            "datasets": list(datasets),
            "protocols": list(PROTOCOLS),
            "budgets": list(BUDGETS),
            "seed": 17,
            "train_scenarios_per_dataset": 1,
            "dev_scenarios_per_dataset": 1,
            "evaluation_turn_budget": 1,
            "official_test_opened": False,
        },
        "gpu": gpu,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "gpu_work_started_at": gpu_work_started_at.isoformat(),
        "gpu_work_ended_at": gpu_work_ended_at.isoformat(),
        "roberta_revision": args.roberta_revision,
        "selected_scenarios": selected,
        "reward_annotation": annotation_reports,
        "reward_annotation_server_compute": annotation_compute,
        "offline_training": offline_reports,
        "selfplay": selfplay_reports,
        "evaluation_report": str(output / "evaluation/mini-evaluation-report.json"),
        "evaluation_dialogues": evaluation["dialogues"],
        "wall_s": time.monotonic() - started,
        "claim_limit": "integration evidence only; not a scientific result",
    }
    atomic_json(output / "mini-e2e-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roberta-revision", required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS))
    parser.add_argument("--runtime-config", type=Path)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output / "mini-e2e-failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "stage": CURRENT_STAGE,
                "error_type": type(error).__name__,
                "error": str(error),
                "official_test_opened": False,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
