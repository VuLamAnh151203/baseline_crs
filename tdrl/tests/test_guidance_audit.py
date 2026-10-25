import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "guidance_audit", Path(__file__).resolve().parents[1] / "scripts/audit_topdial_guidance.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unobserved_future_dialogue_does_not_exempt_reference_topic_copy():
    scenario = {
        "id": "dev",
        "target": ["Movie recommendation", "Current Film"],
        "knowledge": [],
        "profile": {},
        "personality": [],
        "messages": [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Foreign Film"},
            {"role": "user", "content": "I want Foreign Film"},
        ],
    }
    rows = [
        {
            "record": {
                "scenario_id": "dev",
                "protocol": "profile",
                "history": [
                    *scenario["messages"][:2],
                    {"role": "assistant", "content": "Try Foreign Film"},
                    {"role": "user", "content": "No"},
                ],
                "turns_trace": [
                    {
                        "plan": {
                            "reference_id": "train:2",
                            "response": "Try Foreign Film",
                            "path": ["train:2"],
                        }
                    }
                ],
            }
        }
    ]
    refs = {"train:2": {"split": "train", "target": ["Movie recommendation", "Foreign Film"]}}
    result = module.audit(rows, {"dev": scenario}, refs, {"train:2": "recommend"})
    assert len(result["unavailable_reference_topic_flags"]) == 1
    assert result["reference_action_turn_counts"] == {"recommend": 1}
    scenario["knowledge"] = [["Foreign Film", "Genre", "Comedy"]]
    assert not module.audit(rows, {"dev": scenario}, refs, {"train:2": "recommend"})[
        "unavailable_reference_topic_flags"
    ]
