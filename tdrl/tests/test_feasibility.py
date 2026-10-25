import pytest

from tdrl.feasibility import full_turn_dialogue_proxy


def test_early_success_scales_sampling_and_all_other_turn_work():
    # Two measured turns cost 30s total, of which 24s is diffusion sampling.
    # A full 16-turn unsuccessful dialogue needs 240s, including 192s sampling.
    assert full_turn_dialogue_proxy(30, 2, 16) == 240
    assert full_turn_dialogue_proxy(240, 16, 16) == 240


def test_once_per_dialogue_policy_is_not_charged_again_for_every_turn():
    assert full_turn_dialogue_proxy(30, 2, 16, one_time_seconds=24) == 72
    with pytest.raises(ValueError, match="timing/turn"):
        full_turn_dialogue_proxy(30, 2, 16, one_time_seconds=31)


@pytest.mark.parametrize("args", [(1, 0, 16), (1, 17, 16), (float("nan"), 1, 16), (0, 1, 16)])
def test_projection_fails_closed_on_invalid_measurements(args):
    with pytest.raises(ValueError, match="timing/turn"):
        full_turn_dialogue_proxy(*args)


def test_planner_proxy_scales_early_terminal_to_full_declared_recovery_budget():
    from tdrl.feasibility import planner_turn_proxy

    compute = {
        "planner_latency_s": 12,
        "generation_requests": 9,
        "recovery_triggered": True,
        "config": {"candidates": 3, "lookahead": 2, "semantic_rollout_terminals": True},
    }
    assert planner_turn_proxy(compute, 4) == 28


def test_matrix_includes_second_candidate_and_worst_measured_method():
    from tdrl.feasibility import matrix_projection

    methods = {"v8-R": 10, "v8-Full": 30, "v9-R": 10, "v9-Full": 40}
    result = matrix_projection(50, methods, 1e9)
    assert result["required_seconds_with_safety"] == (3664 * 50 + 256 * 40 + 14400) * 1.25
    assert result["fits"]
    assert not matrix_projection(50, methods, 1)["fits"]
    with pytest.raises(ValueError, match="method-specific"):
        matrix_projection(50, {"v8-R": 10}, 1e9)
