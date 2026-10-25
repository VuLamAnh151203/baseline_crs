import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.epl.merge_evaluation import COMMON_FIELDS
from tdrl.baselines.epl.config import EPLConfig
from tdrl.baselines.epl.memory import build_experiences
from tdrl.baselines.epl.rollout import source_rollout_terminal
from tdrl.baselines.epl.search import EPL_SEARCH_REVISION, ExperiencePUCT, puct_score
from tdrl.baselines.epl.training import EPLTrainer
from tdrl.contracts import Message
from tdrl.state import DialogueState

ROOT = Path(__file__).parents[1]


def test_epl_merge_preserves_source_corrected_method_contract():
    assert "method_revision" in COMMON_FIELDS
    assert "method_contract" in COMMON_FIELDS


def test_epl_configs_freeze_deadline_budget():
    topdial = EPLConfig.from_path(ROOT / "configs/epl/topdial.json")
    durecdial = EPLConfig.from_path(ROOT / "configs/epl/durecdial.json")
    assert topdial.dataset == "TopDial"
    assert durecdial.dataset == "DuRecDial2.0"
    for config in (topdial, durecdial):
        assert config.puct_simulations == 5
        assert config.puct_cpuct == 100.0
        assert config.puct_max_depth == 5
        assert config.memory_cases == 376
        assert config.retrieval_k == 20
        assert config.encoder_model == "bert-base-cased"
        assert config.hidden_size == 128
        assert config.retrieval_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.retrieval_dimension == 384
        assert config.training_epochs == 5
        assert config.training_batch_size == 16
        assert config.evaluation_batch_size == 32
        assert config.training_warmup_steps == 6349
        assert config.training_seed == 22
        assert config.self_simulation_seed == 1
        assert len(config.sha256) == len(config.training_sha256) == 64


def test_epl_config_fails_closed_on_unknown_field(tmp_path):
    source = json.loads((ROOT / "configs/epl/topdial.json").read_text())
    source["unfrozen"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="unknown/incomplete"):
        EPLConfig.from_path(path)


def test_build_experiences_reads_training_file_only(tmp_path):
    row = {
        "id": "train:1",
        "profile": {"likes": "science fiction"},
        "personality": {},
        "knowledge": [["Target", "genre", "science fiction"]],
        "target": ["Movie recommendation", "Target"],
        "messages": [
            {"role": "user", "content": "I like science fiction."},
            {"role": "assistant", "content": "I recommend Target."},
            {"role": "user", "content": "I will watch Target."},
        ],
        "assistant_actions": [None, "recommend", None],
        "action_label_source": "weak_rule_v1",
    }
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(row) + "\n")
    output = tmp_path / "experiences.jsonl"
    config = EPLConfig.from_path(ROOT / "configs/epl/topdial.json")
    manifest = build_experiences(source, output, "TopDial", config)
    memory = [json.loads(line) for line in output.read_text().splitlines()]
    assert manifest["source_split"] == "train"
    assert manifest["test_bodies"] == "never_opened"
    assert manifest["experience_count"] == 1
    assert memory[0]["scenario_id"] == "train:1"
    assert memory[0]["continuation"][0]["role"] == "assistant"


@dataclass(frozen=True)
class _Sample:
    assistant_response: str
    reward: float
    terminal: bool
    next_state: tuple
    request_ids: tuple[str, ...]


class _Rollout:
    def expand(self, state, action_index, action_name, path, simulation, depth):
        return _Sample(
            assistant_response=action_name,
            reward=float(action_index),
            terminal=True,
            next_state=tuple(state),
            request_ids=(f"{simulation}/{depth}/{action_name}",),
        )


def test_epl_puct_accounts_for_every_simulation():
    config = EPLConfig.from_path(ROOT / "configs/epl/topdial.json")
    priors = [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
    result = ExperiencePUCT(
        config.actions, lambda state: priors, _Rollout(), config, max_depth=2
    ).search(())
    assert sum(result.root_visits) == config.puct_simulations
    assert result.simulations == config.puct_simulations
    assert 1 <= len(result.request_ids) <= config.puct_simulations
    assert result.action_name in config.actions


def test_epl_source_return_selection_does_not_collapse_on_visit_tie():
    config = EPLConfig.from_path(ROOT / "configs/epl/topdial.json")
    result = ExperiencePUCT(
        config.actions,
        lambda state: [1 / len(config.actions)] * len(config.actions),
        _Rollout(),
        config,
        max_depth=1,
    ).search(())
    assert result.root_visits[:5] == (1, 1, 1, 1, 1)
    assert result.action_index == 4
    assert result.action_name == config.actions[4]
    assert EPL_SEARCH_REVISION == "source-puct-mean-return-v2"


def test_epl_source_rollout_terminal_is_action_based():
    assert source_rollout_terminal("farewell")
    assert not source_rollout_terminal("recommend")
    assert not source_rollout_terminal("acknowledge")


class _NonTerminalRollout:
    def expand(self, state, action_index, action_name, path, simulation, depth):
        assert isinstance(state, DialogueState)
        next_state = state.history + (
            Message("assistant", action_name),
            Message("user", f"reply-{simulation}-{depth}"),
        )
        return _Sample(
            assistant_response=action_name,
            reward=float(action_index),
            terminal=depth > 0,
            next_state=next_state,
            request_ids=(f"{simulation}/{depth}/{action_name}",),
        )


def test_epl_puct_rehydrates_rollout_history_before_deeper_policy_call():
    config = EPLConfig.from_path(ROOT / "configs/epl/topdial.json")
    priors = [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
    observed_history_lengths = []

    def policy(state):
        assert isinstance(state, DialogueState)
        observed_history_lengths.append(len(state.history))
        return priors

    initial = DialogueState((Message("user", "I like science fiction."),))
    result = ExperiencePUCT(
        config.actions, policy, _NonTerminalRollout(), config, max_depth=2
    ).search(initial)
    assert result.max_depth_reached == 2
    assert any(length > len(initial.history) for length in observed_history_lengths)


def test_epl_puct_rejects_invalid_numbers():
    with pytest.raises(ValueError):
        puct_score(float("nan"), 0.5, 1, 0, 1.0)


def test_epl_slurm_jobs_are_separate_and_two_gpu():
    initial = (ROOT / "scripts/slurm/epl_e2e400.sbatch").read_text()
    extension = (ROOT / "scripts/slurm/epl_extend1000.sbatch").read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2" in initial
    assert "--mode initial400" in initial
    assert "--mode extend1000" in extension
    assert "EPL_RUN_ROOT" in extension
    assert "--fixed-test-count" not in initial
    assert "--fixed-test-count" not in extension


def test_epl_comparison_is_natural_only():
    comparison = json.loads((ROOT / "configs/comparison.epl.json").read_text())
    shared = json.loads((ROOT / "configs/baseline-fixed1000-test-v1.json").read_text())
    assert comparison["budgets"] == ["natural"]
    assert comparison["seeds"] == [17]
    assert comparison["protocols"] == shared["evaluation"]["protocols"]
    assert comparison["turn_budget"] == shared["evaluation"]["turn_budget"]


def test_epl_cpu_mini_gate_is_diagnostic_only():
    topdial = EPLConfig.from_path(ROOT / "configs/epl/topdial.json")
    durecdial = EPLConfig.from_path(ROOT / "configs/epl/durecdial.json")
    comparison = json.loads((ROOT / "configs/comparison.epl-mini.json").read_text())
    for config in (topdial, durecdial):
        assert config.training_epochs == 5
        assert config.memory_cases == 376
        assert config.puct_simulations == 5
        assert config.puct_max_depth == 5
    assert comparison["development_scenarios"] == 1
    pipeline = (ROOT / "scripts/epl/run_e2e.py").read_text()
    assert '("--max-train-examples", "4", "--max-dev-examples", "2")' in pipeline
    assert 'command.extend(("--max-cases", "1"))' in pipeline
    batch = (ROOT / "scripts/slurm/epl_cpu_mini.sbatch").read_text()
    assert "#SBATCH --gres" not in batch
    assert "tests/test_epl.py" in batch
    assert "--mode mini" not in batch


def test_epl_training_does_not_shadow_output_directory():
    source = inspect.getsource(EPLTrainer.fit)
    assert "output_dir = Path(output)" in source
    assert 'atomic_json(output_dir / "training-result.json", result)' in source
    assert re.search(r"(?m)^\s*output\s*=\s*self\.model\(", source) is None
