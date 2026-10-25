import json
from pathlib import Path

import pytest

from tdrl.baseline_test_setting import evaluation_axes, load_fixed_test_setting

ROOT = Path(__file__).resolve().parents[1]
SETTING = Path("configs/baseline-fixed1000-test-v1.json")


def test_shared_setting_is_single_seed_natural_budget_and_both_protocols():
    setting = load_fixed_test_setting(ROOT, SETTING)
    assert setting["count"] == 1000
    assert setting["payload"]["test"]["per_dataset"] == {
        "TopDial": 500,
        "DuRecDial2.0": 500,
    }
    assert setting["payload"]["evaluation"] == {
        "seed": 17,
        "protocols": ["profile", "resistant"],
        "turn_budget": 10,
        "budget": "natural",
    }

    config = {
        "seeds": [17, 29, 43],
        "protocols": ["profile", "resistant"],
        "turn_budget": 10,
        "budgets": ["natural", "matched"],
    }
    assert evaluation_axes(setting, config, "dev") == (
        [17, 29, 43],
        ["natural", "matched"],
    )
    assert evaluation_axes(setting, config, "test") == ([17], ["natural"])


@pytest.mark.parametrize("count", [400, 1000])
def test_fixed_setting_supports_only_frozen_prefix_sizes(count):
    setting = load_fixed_test_setting(ROOT, SETTING, count)
    assert setting["count"] == count


def test_modified_shared_setting_and_invalid_prefix_fail_closed(tmp_path):
    original = json.loads((ROOT / SETTING).read_text(encoding="utf-8"))
    original["evaluation"]["budget"] = "matched"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown or modified"):
        load_fixed_test_setting(ROOT, changed)
    with pytest.raises(ValueError, match="must be 400 or 1000"):
        load_fixed_test_setting(ROOT, SETTING, 500)
    with pytest.raises(ValueError, match="requires a fixed-test setting"):
        load_fixed_test_setting(ROOT, None, 400)
