#!/usr/bin/env python3
"""Fail unless every DPDP GPU slot can run concurrently in one allocation."""

import argparse
import json
import os
import time
from pathlib import Path

import torch


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--expected-slots", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    args = parser.parse_args()
    if not 0 <= args.slot < args.expected_slots or args.expected_slots < 2:
        raise ValueError("invalid DPDP parallel-slot probe arguments")
    if args.timeout_seconds <= 0:
        raise ValueError("DPDP parallel-slot timeout must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"DPDP slot {args.slot} requires exactly one visible CUDA device"
        )
    physical_gpu = os.environ.get("SLURM_STEP_GPUS")
    if not physical_gpu:
        raise RuntimeError("SLURM_STEP_GPUS is required for distinct-GPU evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    tensor = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    record = {
        "schema_version": 1,
        "slot": args.slot,
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0),
        "tensor_sum": float(tensor.sum().item()),
        "pid": os.getpid(),
        "ready_monotonic": time.monotonic(),
    }
    atomic_json(args.output / f"ready-{args.slot}.json", record)
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        ready = list(args.output.glob("ready-*.json"))
        if len(ready) == args.expected_slots:
            record["passed_monotonic"] = time.monotonic()
            atomic_json(args.output / f"passed-{args.slot}.json", record)
            return
        time.sleep(0.25)
    raise TimeoutError(
        f"DPDP slot {args.slot} saw only "
        f"{len(list(args.output.glob('ready-*.json')))} of "
        f"{args.expected_slots} concurrent slots"
    )


if __name__ == "__main__":
    main()
