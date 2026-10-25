import json
import subprocess
from pathlib import Path

import pytest

from scripts.dpdp.merge_evaluation import COMMON_FIELDS, merge
from scripts.dpdp.mini_e2e import BUDGETS, DATASETS, PROTOCOLS, _materialize_configs
from scripts.dpdp.run_deadline_e2e import (
    PARALLEL_SLOT_COUNT,
    PORT_BLOCK_START,
    WORKER_MEMORY_GIB,
    DeadlineE2E,
    select_runtime_ports,
)
from scripts.dpdp.run_two_gpu_mini import validate_worker_summaries
from tdrl.baselines.dpdp.cli import SELFPLAY_CHECKPOINT_EVERY
from tdrl.baselines.dpdp.config import DPDPConfig

ROOT = Path(__file__).resolve().parents[1]


def test_dpdp_checkout_stays_clean_if_submission_source_changes(tmp_path):
    repo = tmp_path / "source"
    project = repo / "tdrl"
    project.mkdir(parents=True)
    (repo / ".gitignore").write_text(
        "\n".join(
            (
                "tdrl/data/raw",
                "tdrl/data/processed",
                "tdrl/models",
                "tdrl/toolchains",
                "",
            )
        ),
        encoding="utf-8",
    )
    marker = project / "tracked.txt"
    marker.write_text("pinned\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.name", "DPDP Test"), check=True)
    subprocess.run(
        ("git", "-C", str(repo), "config", "user.email", "dpdp-test@example.invalid"),
        check=True,
    )
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repo), "commit", "-qm", "fixture"), check=True)
    for relative in ("data/raw", "data/processed", "models", "toolchains"):
        (project / relative).mkdir(parents=True)

    helper = ROOT / "scripts/dpdp/prepare_clean_checkout.sh"
    result = subprocess.run(
        ("bash", str(helper), str(project), str(tmp_path / "storage")),
        check=True,
        capture_output=True,
        text=True,
    )
    checkout = Path(result.stdout.strip())
    assert (checkout / "tracked.txt").read_text(encoding="utf-8") == "pinned\n"
    for relative in ("data/raw", "data/processed", "models", "toolchains"):
        link = checkout / relative
        assert link.is_symlink()
        assert link.resolve() == (project / relative).resolve()
    assert not subprocess.check_output(
        ("git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"),
        text=True,
    ).strip()

    marker.write_text("changed after submission\n", encoding="utf-8")
    assert (checkout / "tracked.txt").read_text(encoding="utf-8") == "pinned\n"
    assert not subprocess.check_output(
        ("git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"),
        text=True,
    ).strip()


def test_mini_e2e_configs_are_bounded_and_matched_without_refitting(tmp_path):
    paths = _materialize_configs(ROOT, tmp_path)
    assert tuple(paths) == tuple(DATASETS)
    assert PROTOCOLS == ("profile", "resistant")
    assert BUDGETS == ("natural", "matched")
    for dataset, budgets in paths.items():
        natural = DPDPConfig.from_path(budgets["natural"])
        matched = DPDPConfig.from_path(budgets["matched"])
        assert natural.dataset == dataset
        assert natural.offline_epochs == natural.selfplay_epochs == 1
        assert natural.selfplay_cases_per_epoch == 1
        assert natural.critic_samples == 2
        assert (natural.mcts_simulations, natural.mcts_max_depth) == (3, 2)
        assert (matched.mcts_simulations, matched.mcts_max_depth) == (2, 1)
        assert natural.gate_threshold == matched.gate_threshold == 1.0
        assert natural.training_sha256 == matched.training_sha256
        assert natural.fitting_sha256 != matched.fitting_sha256


def test_mini_e2e_slurm_uses_baseline_physical_storage():
    script = (ROOT / "scripts/slurm/dpdp_mini_e2e.sbatch").read_text(encoding="utf-8")
    assert 'storage_root="/mnt/data/u26anhvl/crs"' in script
    assert 'baseline_name="dpdp"' in script
    assert 'output="${storage_root}/${baseline_name}/dpdp-mini-e2e-${SLURM_JOB_ID}"' in script
    assert 'mkdir -p -- "$output"' in script
    assert 'output="artifacts/' not in script


def test_full_and_smoke_dpdp_slurm_use_same_baseline_storage_root():
    full = (ROOT / "scripts/slurm/dpdp_stage.sbatch").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts/slurm/dpdp_smoke.sbatch").read_text(encoding="utf-8")
    assert 'storage_root="/mnt/data/u26anhvl/crs/dpdp"' in full
    assert 'run_root="${DPDP_RUN_ROOT:-${storage_root}/deadline250-${SLURM_JOB_ID}}"' in full
    assert 'holder="${run_root}/allocation"' in full
    assert 'mkdir -p -- "$holder" "$run_root/logs"' in full
    assert "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2" in full
    assert "#SBATCH --time=6-00:00:00" in full
    assert "scripts/dpdp/run_deadline_e2e.py" in full
    assert 'fixed_test_count="${DPDP_FIXED_TEST_COUNT:-400}"' in full
    assert '--fixed-test-count "$fixed_test_count"' in full
    assert "#SBATCH --signal=B:USR1@900" in full
    assert "#SBATCH --requeue" in full
    assert "gpu-telemetry-${segment_tag}.csv" in full
    assert "controller-${segment_tag}.log" in full
    assert "pipeline_args+=(--resume)" in full
    assert 'storage_root="/mnt/data/u26anhvl/crs/dpdp"' in smoke
    assert 'output="${storage_root}/smoke/dpdp-smoke-${SLURM_JOB_ID}"' in smoke
    assert 'mkdir -p -- "$output"' in smoke


def test_two_gpu_mini_requests_two_h100s_and_disjoint_steps():
    script = (ROOT / "scripts/slurm/dpdp_two_gpu_mini.sbatch").read_text(encoding="utf-8")
    controller = (ROOT / "scripts/dpdp/run_two_gpu_mini.py").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2" in script
    assert 'output="/mnt/data/u26anhvl/crs/dpdp/two-gpu-mini-${SLURM_JOB_ID}"' in script
    assert 'export PYTHONPATH="src:."' in script
    assert '"--exclusive"' in controller
    assert '"--exact"' in controller
    assert '"--gpus-per-task=1"' in controller
    assert '"--gpu-bind=single:1"' in controller
    assert 'f"--mem={WORKER_MEMORY_GIB}G"' in controller


def test_full_dpdp_workers_split_memory_and_require_concurrent_slot_probe():
    command = DeadlineE2E.srun(object(), ["python", "worker.py"])
    assert f"--mem={WORKER_MEMORY_GIB}G" in command
    assert PARALLEL_SLOT_COUNT == 2
    assert WORKER_MEMORY_GIB * PARALLEL_SLOT_COUNT < 160
    source = (ROOT / "scripts/dpdp/run_deadline_e2e.py").read_text(encoding="utf-8")
    assert "self.verify_parallel_slots()" in source
    assert 'self.run_parallel("two-gpu-slot-preflight", commands)' in source


def test_two_gpu_mini_requires_distinct_uuid_and_overlapping_gpu_work():
    def worker(dataset, uuid, start, end):
        return {
            "status": "mini_non_scientific",
            "scope": {"datasets": [dataset], "official_test_opened": False},
            "gpu": {"count": 1, "uuid": uuid},
            "gpu_work_started_at": start,
            "gpu_work_ended_at": end,
        }

    summaries = {
        "TopDial": worker(
            "TopDial", "GPU-a", "2026-10-15T00:00:00+00:00", "2026-10-15T00:10:00+00:00"
        ),
        "DuRecDial2.0": worker(
            "DuRecDial2.0",
            "GPU-b",
            "2026-10-15T00:01:00+00:00",
            "2026-10-15T00:09:00+00:00",
        ),
    }
    evidence = validate_worker_summaries(summaries)
    assert evidence["gpu_work_overlap_s"] == 480
    assert len(set(evidence["distinct_gpu_uuids"].values())) == 2

    summaries["DuRecDial2.0"]["gpu"]["uuid"] = "GPU-a"
    with pytest.raises(ValueError, match="distinct GPU UUIDs"):
        validate_worker_summaries(summaries)


def test_deadline250_configs_preserve_native_training_and_reduce_only_case_count():
    for filename, dataset in (
        ("topdial-deadline250.json", "TopDial"),
        ("durecdial-deadline250.json", "DuRecDial2.0"),
    ):
        config = DPDPConfig.from_path(ROOT / "configs/dpdp" / filename)
        assert config.dataset == dataset
        assert (config.selfplay_epochs, config.selfplay_cases_per_epoch) == (5, 50)
        assert (config.mcts_simulations, config.mcts_max_depth) == (10, 5)
        assert config.critic_samples == 10
        assert (config.offline_dropout, config.selfplay_dropout) == (0.1, 0.25)
        assert config.offline_epochs == 10
        assert config.offline_batch_size == 8
        assert config.offline_lr == 6e-6
        assert config.offline_weight_decay == 0.001
        assert config.offline_warmup_steps == 400
        assert (config.offline_seed, config.selfplay_seed) == (42, 1)
    assert SELFPLAY_CHECKPOINT_EVERY == 5


def test_deadline_resume_archives_stop_markers_and_requires_prior_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "70002")
    root = tmp_path / "project"
    root.mkdir()
    run_root = tmp_path / "deadline250-70001"
    (run_root / "offline" / "TopDial").mkdir(parents=True)
    (run_root / "status.json").write_text("{}", encoding="utf-8")
    (run_root / "stop.requested").touch()
    (run_root / "offline" / "TopDial" / "stop.requested").touch()
    pipeline = DeadlineE2E(root, run_root, Path("/usr/bin/python3"), 400, resume=True)
    pipeline.prepare_run_segment()
    assert not (run_root / "stop.requested").exists()
    assert not (run_root / "offline" / "TopDial" / "stop.requested").exists()
    archive = run_root / "allocation" / "resume-markers" / "70002"
    assert {path.name for path in archive.iterdir()} == {
        "stop.requested",
        "offline__TopDial__stop.requested",
    }
    segment = json.loads((run_root / "allocation/segment-70002.json").read_text())
    assert segment["resume"] is True
    assert len(segment["archived_stop_markers"]) == 2


def test_deadline_fresh_mode_rejects_existing_scientific_evidence(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    run_root = tmp_path / "deadline250-existing"
    run_root.mkdir()
    (run_root / "events.jsonl").write_text("{}\n", encoding="utf-8")
    pipeline = DeadlineE2E(root, run_root, Path("/usr/bin/python3"), 400)
    with pytest.raises(ValueError, match="explicit resume wrapper"):
        pipeline.prepare_run_segment()


def test_dpdp_resume_wrapper_reuses_explicit_physical_run_root():
    submit = (ROOT / "scripts/submit_dpdp_deadline.sh").read_text(encoding="utf-8")
    resume = (ROOT / "scripts/resume_dpdp_deadline.sh").read_text(encoding="utf-8")
    assert 'bash scripts/dpdp/prepare_clean_checkout.sh "$project_root" "$storage_root"' in submit
    assert 'cd "$checkout_project"' in submit
    assert 'resume_run_root="${DPDP_RESUME_RUN_ROOT:-}"' in submit
    assert 'export DPDP_RUN_ROOT="$resume_run_root"' in submit
    assert "export DPDP_RESUME=1" in submit
    assert 'export DPDP_RESUME_RUN_ROOT="$1"' in resume


def test_deadline_runtime_ports_are_job_specific_and_disjoint():
    first = select_runtime_ports("59510", lambda _ports: True)
    second = select_runtime_ports("59511", lambda _ports: True)
    assert first != second
    assert first == {
        "TopDial": (PORT_BLOCK_START + 8_040, PORT_BLOCK_START + 8_041),
        "DuRecDial2.0": (PORT_BLOCK_START + 8_042, PORT_BLOCK_START + 8_043),
    }
    assert len({port for pair in first.values() for port in pair}) == 4


def test_deadline_runtime_ports_skip_an_occupied_job_block():
    inspected = []

    def available(ports):
        inspected.append(ports)
        return len(inspected) == 2

    selected = select_runtime_ports("59510", available)
    assert len(inspected) == 2
    assert selected["TopDial"] == (inspected[1][0], inspected[1][1])
    assert selected["DuRecDial2.0"] == (inspected[1][2], inspected[1][3])


def test_parallel_dataset_reports_merge_to_complete_matrix(tmp_path):
    common = {field: field for field in COMMON_FIELDS}
    common.update(
        {
            "schema_version": 1,
            "status": "complete",
            "split": "dev",
            "method": "DPDP",
            "selection_sha256": "development",
            "evaluation_seeds": [17],
            "evaluation_budgets": ["natural", "matched"],
        }
    )
    paths = []
    for dataset in ("TopDial", "DuRecDial2.0"):
        report = {
            **common,
            "dialogues": 64,
            "wall_s": 2.0,
            "server_compute": {"cold_load_s": 1.0},
            "scope_datasets": [dataset],
            "runtime_config_sha256": dataset,
            "summary": [
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "budget": budget,
                    "method": "DPDP",
                }
                for protocol in ("profile", "resistant")
                for budget in ("natural", "matched")
            ],
            "gate_evidence": {dataset: {}},
        }
        path = tmp_path / f"{dataset}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "merged.json"
    result = merge(paths, output)
    assert result["dialogues"] == 128
    assert len(result["summary"]) == 8
    assert result["scope_datasets"] == ["TopDial", "DuRecDial2.0"]
    assert output.is_file()


def test_parallel_fixed_prefix_reports_merge_natural_only_without_losing_setting(tmp_path):
    common = {field: field for field in COMMON_FIELDS}
    common.update(
        {
            "schema_version": 1,
            "status": "complete",
            "split": "test",
            "method": "DPDP",
            "selection_sha256": "a" * 64,
            "evaluation_seeds": [17],
            "evaluation_budgets": ["natural"],
        }
    )
    test_sample = {
        "requested_prefix_count": 400,
        "selected_counts": {"TopDial": 200, "DuRecDial2.0": 200},
    }
    setting = {
        "scope": "all_baselines_fixed1000_official_test",
        "evaluation": {"seed": 17, "budget": "natural"},
    }
    paths = []
    for dataset in ("TopDial", "DuRecDial2.0"):
        report = {
            **common,
            "dialogues": 400,
            "wall_s": 2.0,
            "server_compute": {"cold_load_s": 1.0},
            "scope_datasets": [dataset],
            "runtime_config_sha256": dataset,
            "summary": [
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "budget": "natural",
                    "method": "DPDP",
                }
                for protocol in ("profile", "resistant")
            ],
            "gate_evidence": {dataset: {}},
            "test_sample": test_sample,
            "dataset_setting": setting,
        }
        path = tmp_path / f"fixed-{dataset}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "fixed-merged.json"
    result = merge(paths, output)
    assert result["dialogues"] == 800
    assert len(result["summary"]) == 4
    assert result["evaluation_budgets"] == ["natural"]
    assert result["test_sample"] == test_sample
    assert result["dataset_setting"] == setting
