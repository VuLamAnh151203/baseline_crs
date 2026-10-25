import json
import subprocess
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from scripts.enpl.smoke import restore_turn_seed_context
from tdrl.backend import final_generation_text
from tdrl.baselines.enpl.bank import build_bank, load_bank
from tdrl.baselines.enpl.consistency import consistency, levenshtein_tokens
from tdrl.baselines.enpl.entity_extractor import DeterministicTextRankExtractor
from tdrl.baselines.enpl.normalize import extract_domain_entities, target_domain
from tdrl.baselines.enpl.planner import ENPL_POLICY_VERSION, EnPLPlanner
from tdrl.baselines.enpl.prompts import render_plan_prompt
from tdrl.baselines.enpl.retrieval import DomainEntityOverlapRetriever
from tdrl.baselines.enpl.schema import Keyphrase, PlanBankEntry
from tdrl.comparison_enpl import validate_config
from tdrl.contracts import Generation, Message, ModelProvenance
from tdrl.state import DialogueState


def test_clean_checkout_is_commit_pinned_and_isolated(tmp_path):
    repo = tmp_path / "repo"
    project = repo / "tdrl"
    project.mkdir(parents=True)
    (repo / ".gitignore").write_text(
        "\n".join(
            (
                "tdrl/data/raw",
                "tdrl/data/processed",
                "tdrl/models",
                "tdrl/toolchains",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    marker = project / "tracked.txt"
    marker.write_text("frozen\n", encoding="utf-8")
    subprocess.run(("git", "init", str(repo)), check=True, capture_output=True)
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=EnPL Test",
            "-c",
            "user.email=enpl@example.invalid",
            "commit",
            "-m",
            "fixture",
        ),
        check=True,
        capture_output=True,
    )
    for relative in ("data/raw", "data/processed", "models", "toolchains"):
        asset = project / relative
        asset.mkdir(parents=True)
        (asset / "asset.txt").write_text(relative, encoding="utf-8")

    helper = Path(__file__).parents[1] / "scripts/enpl/prepare_clean_checkout.sh"
    storage = tmp_path / "storage"
    result = subprocess.run(
        ("bash", str(helper), str(project), str(storage)),
        check=True,
        capture_output=True,
        text=True,
    )
    checkout = Path(result.stdout.strip())
    assert checkout.name == "tdrl"
    assert (checkout / "tracked.txt").read_text(encoding="utf-8") == "frozen\n"
    for relative in ("data/raw", "data/processed", "models", "toolchains"):
        assert (checkout / relative).is_symlink()
        assert (checkout / relative).resolve() == (project / relative).resolve()
    assert (
        subprocess.check_output(
            ("git", "-C", str(checkout), "status", "--porcelain"), text=True
        )
        == ""
    )

    marker.write_text("changed after submission\n", encoding="utf-8")
    assert (checkout / "tracked.txt").read_text(encoding="utf-8") == "frozen\n"
    assert (
        subprocess.check_output(
            ("git", "-C", str(checkout), "status", "--porcelain"), text=True
        )
        == ""
    )


def entry(identifier, group, entities, score=0.8, index=0, domain="movie"):
    return PlanBankEntry(
        entry_id=identifier,
        source_dataset="DuRecDial2.0",
        source_split="train",
        source_conversation_id=group,
        duplicate_group_id=group,
        candidate_index=index,
        plan_text=f"Plan {identifier}",
        user_profile={"Accepted movies": list(entities)},
        goal_raw="goal",
        target_items=("Movie recommendation", next(iter(entities), "Target")),
        demonstration_target=next(iter(entities), "Target"),
        target_resolution="canonical",
        retrieval_domain=domain,
        retrieval_entities=tuple(sorted(entities)),
        conversation_hash=group,
        entity_sequence=("entity",),
        consistency_score=score,
        extraction_config_hash="a" * 64,
        bank_build_config_hash="b" * 64,
    )


def test_entity_sequence_consistency_contract():
    assert levenshtein_tokens(["a", "b"], ["b", "a"]) == 2
    assert consistency(["multi word", "b"], ["multi word", "b"]).score == 1.0
    assert consistency(["a", "b"], ["a", "c"]).score == 0.5
    assert consistency([], ["a"]).valid is False
    assert consistency(["a"], []).reason == "empty_entity_sequence"


def test_audited_enpl_repair_is_exactly_request_scoped():
    request_id = "test/EnPL/profile/17/sample/turn/0/enpl-plan"
    repaired, changed = final_generation_text(
        "<think>1. Ask.\n2. Recommend.</think>",
        "planner",
        request_id,
        request_id,
    )
    assert repaired == "1. Ask.\n2. Recommend."
    assert changed is True
    with pytest.raises(ValueError, match="final text"):
        final_generation_text("<think>hidden</think>", "planner", "other", request_id)


def test_completed_smoke_replay_restores_run_dialogue_seed_context():
    class Planner:
        seed_context = "scenario-only-default"

    planner = Planner()
    restore_turn_seed_context(planner, "scenario-7", "resistant", 0)
    assert planner.seed_context == "scenario-7/resistant/0"
    with pytest.raises(ValueError, match="replay turn"):
        restore_turn_seed_context(planner, "scenario-7", "resistant", -1)


def test_textrank_preserves_adjacent_multiword_phrases():
    phrases = DeterministicTextRankExtractor(keyword_budget=20).extract(
        "Example Film connects Example Film with a thoughtful drama recommendation."
    )
    assert any("example film" in item.canonical_text for item in phrases)
    assert all(
        not (
            "example film" in item.canonical_text
            and "thoughtful" in item.canonical_text
        )
        for item in phrases
    )
    assert all(item.first_start < item.last_end for item in phrases)


def test_domain_entity_retrieval_is_deterministic_same_domain_and_group_distinct():
    entries = [
        entry("a0", "a", {"target film", "other"}, 0.9, 0),
        entry("a1", "a", {"target film"}, 0.8, 1),
        entry("b0", "b", {"target film"}, 0.7, 0),
        entry("c0", "c", {"unrelated"}, 0.9, 0),
        entry("d0", "d", {"another"}, 0.8, 0),
        entry("x0", "x", {"target film"}, 1.0, 0, domain="music"),
    ]
    selected = DomainEntityOverlapRetriever(entries).select(
        {"Accepted movies": ["Target Film"]},
        ("Movie recommendation", "Target Film"),
    )
    assert [item.entry.entry_id for item in selected] == ["a0", "b0", "c0"]
    assert [item.overlap_score for item in selected] == [1, 1, 0]
    assert selected[-1].zero_overlap_fallback is True


class FakeExtractor:
    config = {"implementation": "fake_for_test"}
    config_hash = "f" * 64

    def extract(self, text):
        words = [item for item in text.casefold().split() if item.startswith("e")]
        return [
            Keyphrase(item, item, 1.0, index, index + len(item))
            for index, item in enumerate(words)
        ]


def test_bank_uses_scores_and_only_exact_mapped_training_records(tmp_path):
    processed = tmp_path / "data/processed/DuRecDial2.0"
    processed.mkdir(parents=True)
    scenario = {
        "id": "train:en:000000",
        "messages": [{"role": "user", "content": "entity a entity b"}],
        "target": ["Movie recommendation", "Target Film"],
    }
    (processed / "train.jsonl").write_text(json.dumps(scenario) + "\n")
    (processed / "dev.jsonl").write_text("")
    record = {
        "conver_id": 9,
        "plans": ["x"] * 8 + ["entity a", "entity a entity b"],
        "user_profile": {"Accepted movies": ["Target Film"]},
        "conversation": ["entity a entity b"],
        "goal": "[1] Movie recommendation(Target Film)",
    }
    archive = tmp_path / "ConvPlan.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("ConvPlan.json", json.dumps([record]))
    output = tmp_path / "bank"
    manifest = build_bank(
        tmp_path, archive, output, max_records=1, extractor=FakeExtractor()
    )
    assert manifest["status"] == "diagnostic_partial"
    entries, _ = load_bank(output / "plan-bank.jsonl", output / "bank-manifest.json", False)
    assert [item.candidate_index for item in entries] == [8, 9]
    scores = {item.candidate_index: item.consistency_score for item in entries}
    assert scores[9] > scores[8]
    assert all(item.source_split == "train" for item in entries)


class FakeGeneration:
    def __init__(self):
        self.requests = []

    def generate(self, messages, role, decoding, seed, request_id):
        text = (
            "1. Acknowledge the user's preference.\n2. Bridge naturally toward Target Film."
            if role == "planner"
            else "Try Target Film."
        )
        result = Generation(
            text=text,
            input_tokens=20,
            output_tokens=8,
            total_tokens=28,
            seed=seed,
            decoding=decoding,
            latency_s=0.1,
            retries=0,
            model=ModelProvenance(
                "Qwen/Qwen3.5-9B", "a" * 40, {"weights": "b" * 64}, "c" * 64
            ),
            prompt_sha256="d" * 64,
            request_id=request_id,
            role=role,
            finish_reason="stop",
        )
        self.requests.append((messages, result))
        return result


class MalformedPlanGeneration(FakeGeneration):
    def generate(self, messages, role, decoding, seed, request_id):
        result = super().generate(messages, role, decoding, seed, request_id)
        if role == "planner":
            return replace(result, text="A direct conversational reply.")
        return result


class BulletedPlanGeneration(FakeGeneration):
    def generate(self, messages, role, decoding, seed, request_id):
        result = super().generate(messages, role, decoding, seed, request_id)
        if role == "planner":
            return replace(
                result,
                text=(
                    "- Acknowledge the user's preference.\n"
                    "- Bridge naturally toward Target Film."
                ),
            )
        return result


class LongTruncatedPlanGeneration(FakeGeneration):
    def generate(self, messages, role, decoding, seed, request_id):
        result = super().generate(messages, role, decoding, seed, request_id)
        if role == "planner":
            return replace(
                result,
                text="\n".join(
                    f"{index}. Future conversation step {index}."
                    for index in range(1, 15)
                ),
                finish_reason="length",
            )
        return result


def test_planner_replans_then_generates_one_response_without_hidden_future():
    scenario = {
        "id": "s1",
        "target": ["Movie recommendation", "Target Film"],
        "profile": {"Accepted movies": ["Target Film"]},
        "knowledge": [["Target Film", "genre", "drama"]],
        "personality": "HIDDEN_PERSONALITY",
        "messages": [
            {"role": "user", "content": "Observed request"},
            {"role": "assistant", "content": "HIDDEN_FUTURE"},
        ],
    }
    examples = [entry(str(i), str(i), {"target film"}) for i in range(3)]
    backend = FakeGeneration()
    planner = EnPLPlanner(backend, scenario, DomainEntityOverlapRetriever(examples))
    plan = planner.act(
        DialogueState((Message("user", "Observed request"),)), [], 17, "enpl/turn/0"
    )
    assert [request.role for _messages, request in backend.requests] == ["planner", "assistant"]
    assert plan.response == "Try Target Film."
    assert plan.compute["policy_version"] == ENPL_POLICY_VERSION
    assert plan.compute["generation_requests"] == 2
    assert plan.compute["examples_used"] == 3
    assert plan.compute["plan_format_valid"] is True
    assert plan.compute["plan_steps"] == 2
    serialized = json.dumps(asdict(plan))
    prompts = json.dumps(
        [[asdict(message) for message in messages] for messages, _ in backend.requests]
    )
    assert "HIDDEN_PERSONALITY" not in serialized + prompts
    assert "HIDDEN_FUTURE" not in serialized + prompts


def test_planner_drops_lowest_ranked_whole_examples_for_context():
    scenario = {
        "id": "bounded",
        "target": ["Movie recommendation", "Target Film"],
        "profile": {"Accepted movies": ["Target Film"]},
        "knowledge": [],
        "messages": [{"role": "user", "content": "Observed request"}],
    }
    examples = [
        PlanBankEntry(**{**asdict(entry(str(i), str(i), {"target film"})), "plan_text": "x" * 900})
        for i in range(3)
    ]
    backend = FakeGeneration()
    planner = EnPLPlanner(
        backend,
        scenario,
        DomainEntityOverlapRetriever(examples),
        max_input_characters=3000,
    )
    plan = planner.act(
        DialogueState((Message("user", "Observed request"),)), [], 17, "bounded/turn/0"
    )
    assert plan.compute["examples_requested"] == 3
    assert 1 <= plan.compute["examples_used"] < 3
    assert plan.compute["examples_dropped_for_context"] > 0
    assert plan.compute["plan_input_characters"] <= 3000


def test_planner_can_drop_all_examples_while_preserving_observed_history():
    scenario = {
        "id": "zero-example-context-fallback",
        "target": ["Movie recommendation", "Target Film"],
        "profile": {"Accepted movies": ["Target Film"]},
        "knowledge": [["Target Film", "genre", "drama"]],
        "messages": [{"role": "user", "content": "Observed request"}],
    }
    history = (Message("user", "Observed request"),)
    examples = [
        PlanBankEntry(
            **{
                **asdict(entry(str(index), str(index), {"target film"})),
                "plan_text": "x" * 900,
            }
        )
        for index in range(3)
    ]
    zero_example_prompt = render_plan_prompt(scenario, [])
    budget = len(zero_example_prompt) + len(history[0].content)
    retriever = DomainEntityOverlapRetriever(examples)
    retrieved = retriever.select(scenario["profile"], scenario["target"])
    planner = EnPLPlanner(
        FakeGeneration(),
        scenario,
        retriever,
        max_input_characters=budget,
    )

    prompt, used = planner._bounded_plan_prompt(DialogueState(history), retrieved)

    assert prompt == zero_example_prompt
    assert used == ()
    assert len(prompt) + len(history[0].content) == budget


def test_planner_records_non_plan_output_without_hidden_recovery():
    scenario = {
        "id": "malformed",
        "target": ["Movie recommendation", "Target Film"],
        "profile": {"Accepted movies": ["Target Film"]},
        "knowledge": [],
    }
    examples = [entry(str(i), str(i), {"target film"}) for i in range(3)]
    backend = MalformedPlanGeneration()
    planner = EnPLPlanner(backend, scenario, DomainEntityOverlapRetriever(examples))
    plan = planner.act(
        DialogueState((Message("user", "Observed request"),)),
        [],
        17,
        "malformed/turn/0",
    )
    assert plan.compute["plan_format"] == "free_text"
    assert plan.compute["plan_format_valid"] is False
    assert plan.compute["plan_format_violation"] is True
    assert plan.compute["generation_requests"] == 2
    assert plan.compute["recovery_triggered"] is False
    assert len(backend.requests) == 2


def test_planner_accepts_bulleted_multi_step_plan():
    scenario = {
        "id": "bulleted",
        "target": ["Movie recommendation", "Target Film"],
        "profile": {"Accepted movies": ["Target Film"]},
        "knowledge": [],
    }
    examples = [entry(str(i), str(i), {"target film"}) for i in range(3)]
    planner = EnPLPlanner(
        BulletedPlanGeneration(), scenario, DomainEntityOverlapRetriever(examples)
    )
    plan = planner.act(
        DialogueState((Message("user", "Observed request"),)),
        [],
        17,
        "bulleted/turn/0",
    )
    assert plan.compute["policy_version"] == (
        "independent-enpl-domain3-history-conditioned-v4"
    )
    assert plan.compute["plan_format"] == "bulleted"
    assert plan.compute["plan_format_valid"] is True
    assert plan.compute["plan_format_violation"] is False
    assert plan.compute["plan_steps"] == 2


def test_planner_accepts_and_discloses_long_truncated_multi_step_plan():
    scenario = {
        "id": "long-plan",
        "target": ["Movie recommendation", "Target Film"],
        "profile": {"Accepted movies": ["Target Film"]},
        "knowledge": [],
    }
    examples = [entry(str(i), str(i), {"target film"}) for i in range(3)]
    planner = EnPLPlanner(
        LongTruncatedPlanGeneration(),
        scenario,
        DomainEntityOverlapRetriever(examples),
    )
    plan = planner.act(
        DialogueState((Message("user", "Observed request"),)),
        [],
        17,
        "long-plan/turn/0",
    )
    assert plan.compute["plan_format"] == "numbered"
    assert plan.compute["plan_format_valid"] is True
    assert plan.compute["plan_steps"] == 14
    assert plan.compute["plan_finish_reason"] == "length"
    assert plan.compute["plan_truncated"] is True
    assert plan.compute["generation_requests"] == 2


def test_enpl_configs_keep_shared_axes():
    root = Path(__file__).parents[1]
    full = json.loads((root / "configs/comparison.enpl.json").read_text())
    mini = json.loads((root / "configs/comparison.enpl-mini.json").read_text())
    validate_config(full)
    validate_config(mini)
    assert full["development_scenarios"] == 16
    assert mini["development_scenarios"] == 1
    assert extract_domain_entities(
        {"Accepted movies": ["A"], "Accepted music": ["B"]},
        ("Movie recommendation", "C"),
    ) == ("movie", ("a", "c"))
    assert extract_domain_entities(
        {"Accepted movies": ["A"], "Accepted music": ["B; D"]},
        ("Play music", "C"),
    ) == ("music", ("b", "c", "d"))
    assert target_domain(("POI recommendation", "Cafe")) == "poi"
