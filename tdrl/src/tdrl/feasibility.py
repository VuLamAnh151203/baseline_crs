"""Conservative warm-dialogue extrapolation, including intrinsic policy sampling."""

import math


def full_turn_dialogue_proxy(wall_seconds, actual_turns, turn_budget, one_time_seconds=0):
    if (
        type(wall_seconds) not in {int, float}
        or not math.isfinite(wall_seconds)
        or wall_seconds <= 0
        or type(actual_turns) is not int
        or type(turn_budget) is not int
        or not 0 < actual_turns <= turn_budget
        or type(one_time_seconds) not in {int, float}
        or not math.isfinite(one_time_seconds)
        or not 0 <= one_time_seconds <= wall_seconds
    ):
        raise ValueError("invalid measured dialogue timing/turn count")
    return one_time_seconds + (wall_seconds - one_time_seconds) * turn_budget / actual_turns


def planner_turn_proxy(compute, external_seconds):
    latency = compute["planner_latency_s"]
    calls = compute["generation_requests"]
    if (
        not math.isfinite(latency)
        or latency <= 0
        or not math.isfinite(external_seconds)
        or external_seconds < 0
        or calls <= 0
    ):
        raise ValueError("invalid measured planner timing/calls")
    cfg = compute["config"]
    declared = (
        (cfg["candidates"] * cfg["lookahead"] * (3 if cfg["semantic_rollout_terminals"] else 2))
        if compute["recovery_triggered"]
        else 1
    )
    if calls > declared:
        raise ValueError("measured planner exceeded declared generation budget")
    return latency * declared / calls + external_seconds


def matrix_projection(
    difftod_seconds,
    planner_seconds,
    remaining_seconds,
    reserve=3664,
    extra_candidate_dialogues=256,
    io_reserve=14400,
    safety=1.25,
):
    values = [difftod_seconds, remaining_seconds, *planner_seconds.values()]
    if any(not math.isfinite(x) or x <= 0 for x in values) or set(planner_seconds) != {
        "v8-R",
        "v8-Full",
        "v9-R",
        "v9-Full",
    }:
        raise ValueError("complete finite method-specific measurements required")
    worst = max(difftod_seconds, *planner_seconds.values())
    extra_worst = max(planner_seconds["v9-R"], planner_seconds["v9-Full"])
    required = (reserve * worst + extra_candidate_dialogues * extra_worst + io_reserve) * safety
    return {
        "required_seconds_with_safety": required,
        "remaining_allocation_seconds": remaining_seconds,
        "fits": required <= remaining_seconds,
        "original_dialogue_reserve": reserve,
        "extra_candidate_dialogue_reserve": extra_candidate_dialogues,
        "worst_common_dialogue_proxy_seconds": worst,
        "method_dialogue_proxy_seconds": {"DiffTOD": difftod_seconds, **planner_seconds},
    }
