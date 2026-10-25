"""Allocation routing control tests; no model inference or scheduler mutation."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_in_allocation.py"
spec = importlib.util.spec_from_file_location("allocation_step", SCRIPT)
allocation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(allocation)

RECORD = (
    "JobId=123 UserId=duclm3(1132) JobName=tdrl-persistent-allocation "
    "JobState=RUNNING NumCPUs=16 TimeLimit=20-00:00:00 "
    "AllocTRES=cpu=16,mem=128G,node=1,gres/gpu=1 "
    "TresPerNode=gres/gpu:nvidia_h100_80gb_hbm3:1"
)


def test_allocation_contract_accepts_only_owned_running_requested_resources():
    allocation.validate_allocation(RECORD, 1132)
    for old, new in [
        ("1132", "999"),
        ("RUNNING", "PENDING"),
        ("mem=128G", "mem=160G"),
        ("NumCPUs=16", "NumCPUs=8"),
        ("20-00:00:00", "21-00:00:00"),
        ("nvidia_h100_80gb_hbm3", "nvidia_h100_80gb_hbm3_3g.40gb"),
        ("tdrl-persistent-allocation", "unknown-job"),
    ]:
        with pytest.raises(ValueError):
            allocation.validate_allocation(RECORD.replace(old, new), 1132)


def test_followup_steps_reserve_entire_allocation_without_overlap_or_new_job():
    command = allocation.step_command("123", ["runtime/.venv/bin/python", "-m", "tdrl", "check"])
    assert command[0] == "srun"
    assert "--jobid=123" in command and "--cpus-per-task=16" in command
    assert "--gres=gpu:nvidia_h100_80gb_hbm3:1" in command
    assert "--overlap" not in command
    with pytest.raises(ValueError):
        allocation.step_command("123; bad", ["anything"])


def test_two_gpu_resource_contract_and_one_gpu_eight_cpu_workers():
    record = RECORD.replace("gres/gpu=1", "gres/gpu=2").replace("hbm3:1", "hbm3:2")
    allocation.validate_allocation(record, 1132, gpu_count=2)
    with pytest.raises(ValueError):
        allocation.validate_allocation(record, 1132)
    cmd = allocation.step_command("123", ["worker"], worker_cpus=8)
    assert "--cpus-per-task=8" in cmd
    assert "--gres=gpu:nvidia_h100_80gb_hbm3:1" in cmd
