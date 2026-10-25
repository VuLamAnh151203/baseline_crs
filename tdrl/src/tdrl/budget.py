import math

METHODS = [
    "qwen",
    "tdrl_r",
    "tdrl",
    "cosine",
    "random_routing",
    "always_rollout",
    "PPDPP",
    "DPDP",
    "TRIP",
    "DiffTOD",
]
ADAPTIVE = METHODS[2:]


def experiment_matrix(splits: dict[str, list[str]], seeds=(17, 29, 43)):
    if set(splits) != {"TopDial", "DuRecDial2.0"} or len(set(seeds)) != 3:
        raise ValueError("both datasets and three distinct seeds required")
    matrix = []
    for dataset, split_names in splits.items():
        for split in split_names:
            for protocol in ("profile", "resistant"):
                for seed in seeds:
                    for method in METHODS:
                        budgets = ("natural", "matched") if method in ADAPTIVE else ("natural",)
                        for budget in budgets:
                            matrix.append(
                                {
                                    "dataset": dataset,
                                    "split": split,
                                    "protocol": protocol,
                                    "seed": seed,
                                    "method": method,
                                    "budget": budget,
                                }
                            )
    return matrix


def feasibility(
    cells: list[dict],
    counts: dict,
    seconds_per_dialogue: dict,
    remaining_hours: float,
    reserve_hours=24,
    safety_factor=1.25,
):
    """Use measured warm dialogue p95 by cell; absence is not a throughput estimate."""
    if not 0 < remaining_hours <= 480 or reserve_hours < 0 or safety_factor < 1:
        raise ValueError("invalid allocation budget")
    total = 0.0
    for cell in cells:
        count_key = f"{cell['dataset']}/{cell['split']}"
        timing_key = f"{cell['method']}/{cell['budget']}/{cell['protocol']}/{cell['dataset']}"
        count, seconds = counts[count_key], seconds_per_dialogue[timing_key]
        if count <= 0 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("positive audited counts and measured timing required")
        total += count * seconds
    required = total * safety_factor / 3600
    available = max(0, remaining_hours - reserve_hours)
    return {
        "estimated_hours": required,
        "available_hours": available,
        "safety_factor": safety_factor,
        "fits": required <= available,
        "status": "estimate_not_completion",
    }
