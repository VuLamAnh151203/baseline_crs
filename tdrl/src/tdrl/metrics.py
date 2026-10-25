import math

import numpy as np


def paired_bootstrap(left, right, seed=20261003, resamples=10000):
    """Scenario clusters retain all configured seeds; paired IDs across methods."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim not in {1, 2} or len(a) < 2:
        raise ValueError("paired scenario arrays required")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or resamples != 10000:
        raise ValueError("finite values and 10,000 resamples required")
    delta = (a - b).reshape(len(a), -1).mean(axis=1)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples)
    for offset in range(0, resamples, 100):
        idx = rng.integers(0, len(a), size=(min(100, resamples - offset), len(a)))
        samples[offset : offset + len(idx)] = delta[idx].mean(axis=1)
    return {
        "difference": float(delta.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "resamples": resamples,
        "seed": seed,
        "unit": "scenario_cluster",
    }


def validate_records(records, scenario_ids, seeds, lock_hash):
    expected = {(s, seed) for s in scenario_ids for seed in seeds}
    found = set()
    for r in records:
        key = (r["scenario_id"], r["seed"])
        if key in found or key not in expected:
            raise ValueError("duplicate/unexpected result")
        found.add(key)
        if r["selection_sha256"] != lock_hash or r["infrastructure_errors"] != 0:
            raise ValueError("lock mismatch/unresolved infrastructure errors")
        if not isinstance(r["accepted"], bool):
            raise ValueError("deterministic acceptance label required")
        for name in (
            "turns",
            "generation_requests",
            "embedding_requests",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "h100_seconds",
            "wall_s",
        ):
            if not math.isfinite(r[name]) or r[name] < 0:
                raise ValueError("invalid result counter")
        if r["total_tokens"] != r["input_tokens"] + r["output_tokens"]:
            raise ValueError("token total mismatch")
        if not r["accepted"] and r["turns"] != r["turn_budget"]:
            raise ValueError("unsuccessful dialogues count full turn budget")
    if found != expected:
        raise ValueError("partial result matrix")


def aggregate(records):
    if not records:
        raise ValueError("empty results")
    successes = sum(r["accepted"] for r in records)
    compute = sum(r["h100_seconds"] for r in records)
    result = {
        "dialogues": len(records),
        "successes": successes,
        "sr": successes / len(records),
        "average_turns": np.mean([r["turns"] for r in records]).item(),
        "h100_seconds_per_dialogue": compute / len(records),
        "h100_seconds_per_success": compute / successes if successes else None,
    }
    if all("outcome_label" in r for r in records):
        result["recommendation_acceptances"] = sum(
            r["outcome_label"] == "acceptance" for r in records
        )
        result["other_goal_completions"] = sum(
            r["outcome_label"] == "goal_completed" for r in records
        )
        result["goal_type_results"] = {
            kind: {
                "dialogues": sum(r["goal_type"] == kind for r in records),
                "successes": sum(r["accepted"] for r in records if r["goal_type"] == kind),
            }
            for kind in sorted({r["goal_type"] for r in records})
        }
    return result
