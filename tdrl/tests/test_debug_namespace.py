import importlib.util
from pathlib import Path

import pytest

from tdrl.evaluation import dialogue_request_base


def test_debug_namespace_distinguishes_head_variant_threshold_and_code():
    path = Path(__file__).resolve().parents[1] / "scripts/debug_topdial_planner.py"
    spec = importlib.util.spec_from_file_location("debug_planner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = ("repair-MRNFixed-R", ["node"], {"commit": "source"}, "config", "mrn", {"H_cover": 1})
    contracts = [module.diagnostic_contract(*base)]
    for i, value in [
        (0, "repair-IQE-R"),
        (4, "different-head"),
        (5, {"H_cover": 2}),
        (2, {"commit": "new"}),
    ]:
        args = list(base)
        args[i] = value
        contracts.append(module.diagnostic_contract(*args))
    ids = [dialogue_request_base(c, "profile", 17, "same-scenario") for c in contracts]
    assert len(ids) == len(set(ids))
    assert module.diagnostic_contract(*base) == contracts[0]


def test_targetpool_smoke_covers_available_anchor_and_unseen_target():
    path = Path(__file__).resolve().parents[1] / "scripts/debug_topdial_planner.py"
    spec = importlib.util.spec_from_file_location("debug_planner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = {"reference_scope": "exact_target_when_supported"}
    cases = [{"target": ["movie", "Unseen"]}, {"target": ["movie", "Known"]}]
    records = [
        {"split": "train", "guidance": "next", "goal_reference": True,
         "target": ["movie", "Known"]},
        {"split": "dev", "guidance": "next", "goal_reference": True,
         "target": ["movie", "Unseen"]},
    ]
    assert module.smoke_fixtures(cfg, cases, records) == [
        ("exact_target", cases[1]), ("goal_type_transfer", cases[0])
    ]
    with pytest.raises(ValueError, match="exact-anchor and transfer"):
        module.smoke_fixtures(cfg, cases[:1], records)
