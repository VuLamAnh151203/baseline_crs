"""Shared interactive simulation, deterministic labels, and resumable dialogue records."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .acceptance import classify_user
from .autonomy import explicit_refusal
from .contracts import Decoding, Message, Plan
from .planner import ASSISTANT_POLICY, derived_seed
from .provenance import atomic_json, digest
from .state import DialogueState
from .strategies import ACTION_INSTRUCTIONS

USER_POLICY = (
    "Role-play the user whose profile and personality are supplied. Respond naturally to the "
    "assistant based on your own preferences. You may be skeptical, refuse, correct the assistant, "
    "or change topics. Do not accept simply because a target is supplied. Only if you independently "
    "choose an item, name it explicitly and state your intended action. Mention, uncertainty, "
    "prior consumption and polite thanks are not a new acceptance. Profile Accepted/Rejected "
    "entries describe prior preferences, not an exhaustive whitelist of future choices. Do not "
    "infer that you have already consumed an item merely because it is in an Accepted field. "
    "Keep stated facts and preferences consistent across turns; do not invent prior consumption "
    "to contradict them. Refusal remains a valid outcome. Keep replies to one or two short "
    "sentences. Return only your final reply."
)
RESISTANT_POLICY = (
    "Be resistant to unsolicited persuasion: ask for reasons, challenge mismatched claims, "
    "and decline recommendations that do not suit your profile. Accept only if the conversation "
    "has persuaded you without pressure. Refusal remains a valid outcome."
)


def visible_context(scenario):
    return json.dumps(
        {
            "target": scenario["target"],
            "profile": scenario["profile"],
            "knowledge": scenario["knowledge"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def simulator_prompt(scenario, protocol):
    if protocol not in {"profile", "resistant"}:
        raise ValueError("unknown frozen user protocol")
    return (
        USER_POLICY
        + ("\n" + RESISTANT_POLICY if protocol == "resistant" else "")
        + "\n"
        + json.dumps(
            {
                "profile": scenario["profile"],
                "personality": scenario["personality"],
                "permitted_knowledge": scenario["knowledge"],
                "system_target": scenario["target"],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def initial_state(scenario):
    prefix = []
    for row in scenario["messages"]:
        prefix.append(Message(**row))
        if row["role"] == "user":
            break
    if not prefix or prefix[-1].role != "user":
        raise ValueError("scenario has no legitimate initial user observation")
    return DialogueState(tuple(prefix))


def user_reply(backend, state, prompt, decoding, seed, request_id):
    flipped = [
        Message("assistant" if m.role == "user" else "user", m.content) for m in state.history
    ]
    return backend.generate(
        [Message("system", prompt), *flipped], "simulator", decoding, seed, request_id
    )


def policy_input(tokenizer, state, context, max_tokens=512):
    """Documented task adaptation: keep a bounded target header plus recent whole turns."""
    supplied = json.loads(context)
    header_text = json.dumps(
        {"target": supplied["target"], "profile": supplied["profile"]}, ensure_ascii=False
    )
    header = tokenizer.encode(header_text, add_special_tokens=False)[:96]
    remaining = max_tokens - len(header) - 2
    selected = []
    for turn in reversed(state.history):
        tokens = tokenizer.encode(f"{turn.role}: {turn.content}", add_special_tokens=False)
        if len(tokens) > remaining:
            break
        selected = tokens + selected
        remaining -= len(tokens)
    ids = [tokenizer.cls_token_id] + header + selected + [tokenizer.sep_token_id]
    tensor = torch.tensor([ids], dtype=torch.long, device="cuda")
    return tensor, torch.ones_like(tensor)


class PPDPPPlanner:
    def __init__(
        self,
        policy,
        tokenizer,
        generation,
        embedding,
        context,
        decoding,
        training=False,
        policy_seed=17,
    ):
        self.policy, self.tokenizer = policy, tokenizer
        self.generation, self.embedding = generation, embedding
        self.context, self.decoding, self.training = context, decoding, training
        self.sample_generator = torch.Generator(device="cuda").manual_seed(policy_seed)
        self.log_probabilities = []

    def act(self, state, target_nodes, seed, request_id):
        from .autonomy import REFUSAL_RESPONSE
        from .planner import INSTRUCTION

        self.embedding.embed(
            [state.serialize()], INSTRUCTION, request_id + "/live", request_id + "/live"
        )
        if not getattr(self, "semantic_judging", False) and explicit_refusal(state):
            return Plan(
                REFUSAL_RESPONSE,
                False,
                None,
                {},
                [],
                None,
                [],
                {"refusal_guard": True, "recovery_triggered": False},
                [],
            )
        ids, mask = policy_input(self.tokenizer, state, self.context)
        with torch.set_grad_enabled(self.training):
            action, log_prob = self.policy.select(
                ids, mask, self.sample_generator, evaluate=not self.training
            )
        if self.training:
            self.log_probabilities.append(log_prob.squeeze())
        strategy = self.policy.actions[int(action.item())]
        reply = self.generation.generate(
            [
                Message(
                    "system",
                    ASSISTANT_POLICY
                    + "\n"
                    + self.context
                    + "\nNext strategy: "
                    + strategy
                    + "\n"
                    + ACTION_INSTRUCTIONS[strategy],
                ),
                *state.history,
            ],
            "assistant",
            self.decoding,
            derived_seed(seed, self.seed_context, "final"),
            request_id + "/final",
        )
        return Plan(
            reply.text,
            False,
            None,
            {},
            [],
            None,
            [],
            {"strategy": strategy, "recovery_triggered": False},
            [reply.request_id],
        )


def dialogue_request_base(contract, protocol, seed, scenario_id):
    base = f"{contract['phase']}/{contract['method']}/{protocol}/{seed}/{scenario_id}"
    namespace = contract.get("request_namespace")
    if namespace is not None:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("invalid dialogue request namespace")
        base += "/namespace/" + digest(namespace)
    return base


def run_dialogue(
    scenario,
    planner,
    generation,
    embedding,
    protocol,
    seed,
    output: Path,
    contract: dict,
    turn_budget=10,
    simulator_decoding=None,
    replay=False,
    interrupt_after=None,
    judge=None,
    training_critic=False,
):
    simulator_decoding = simulator_decoding or Decoding(max_tokens=128)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "dialogue.json"
    if training_critic and judge is None:
        raise ValueError("sampled training feedback requires local judge")
    contract = {**contract, "judge_sha256": judge.protocol_hash} if judge else contract
    contract_hash = digest(contract)
    if snapshot_path.exists():
        stored = json.loads(snapshot_path.read_text())
        if stored["contract_sha256"] != contract_hash:
            raise ValueError("dialogue resume contract mismatch")
        if stored["status"] == "complete":
            return stored
        state = DialogueState(tuple(Message(**m) for m in stored["history"]))
        trace = stored["turns_trace"]
        start_turn = len(trace)
        result = stored
    else:
        state, trace, start_turn = initial_state(scenario), [], 0
    base = dialogue_request_base(contract, protocol, seed, scenario["id"])
    prompt = simulator_prompt(scenario, protocol)
    consumed = getattr(generation, "consumed_requests", generation.requests)
    started = time.monotonic()
    accepted = result["accepted"] if start_turn else False
    label = None
    planner.semantic_judging = judge is not None
    terminal = accepted or (
        start_turn > 0
        and (trace[-1]["acceptance"]["refused"] if judge else explicit_refusal(state))
    )
    for turn in [] if terminal else range(start_turn, turn_budget):
        seed_context = f"{scenario['id']}/{protocol}/{turn}"
        planner.seed_context = seed_context
        planner.simulator_prompt = prompt
        gstart, estart = len(consumed), len(embedding.requests)
        tstart = time.monotonic()
        plan = planner.act(state, contract.get("target_nodes", []), seed, base + f"/turn/{turn}")
        state = state.append(Message("assistant", plan.response))
        user = user_reply(
            generation,
            state,
            prompt,
            simulator_decoding,
            derived_seed(seed, seed_context, "simulator"),
            base + f"/turn/{turn}/user",
        )
        state = state.append(Message("user", user.text))
        label = (
            judge.assess(
                state,
                {**scenario, "latest_user_reply_finish_reason": user.finish_reason},
                derived_seed(seed, seed_context, "judge"),
                base + f"/turn/{turn}/judge",
            )
            if judge
            else classify_user(state, scenario["target"])
        )
        feedback = (
            judge.feedback(
                state,
                {**scenario, "latest_user_reply_finish_reason": user.finish_reason},
                seed,
                base + f"/turn/{turn}/critic",
            )
            if training_critic
            else None
        )
        accepted = label.accepted
        # Other dialogue workers share the server; attribute only this turn's IDs.
        prefix = base + f"/turn/{turn}/"
        actual_gen = [r for r in consumed[gstart:] if r.request_id.startswith(prefix)]
        actual_emb = [r for r in embedding.requests[estart:] if r.request_id.startswith(prefix)]
        trace.append(
            {
                "turn": turn,
                "plan": asdict(plan),
                "acceptance": asdict(label),
                "critic": feedback,
                "generation": [asdict(r) for r in actual_gen],
                "embedding": [
                    {
                        "request_id": r.request_id,
                        "batch_size": r.batch_size,
                        "input_tokens": r.input_tokens,
                        "latency_s": r.latency_s,
                        "vectors_sha256": digest(r.vectors.tolist()),
                    }
                    for r in actual_emb
                ],
                "wall_s": time.monotonic() - tstart,
                "simulator_seed": user.seed,
            }
        )
        result = {
            "schema_version": 1,
            "status": "partial",
            "contract_sha256": contract_hash,
            "scenario_id": scenario["id"],
            "seed": seed,
            "protocol": protocol,
            "method": contract["method"],
            "history": [asdict(m) for m in state.history],
            "turns_trace": trace,
            "accepted": accepted,
        }
        atomic_json(snapshot_path, result)
        if interrupt_after is not None and len(trace) >= interrupt_after:
            return result
        if accepted or (label.refused if judge else explicit_refusal(state)):
            break
    if not trace:
        raise ValueError("dialogue has no executed turns")
    generations = [r for t in trace for r in t["generation"]]
    embeddings = [r for t in trace for r in t["embedding"]]
    elapsed = sum(t["wall_s"] for t in trace)
    result.update(
        status="complete",
        actual_turns=len(trace),
        turns=len(trace) if accepted else turn_budget,
        turn_budget=turn_budget,
        selection_sha256=contract.get("selection_sha256", "development_or_smoke"),
        infrastructure_errors=0,
        generation_requests=len(generations),
        embedding_requests=len(embeddings),
        input_tokens=sum(r["input_tokens"] for r in generations),
        output_tokens=sum(r["output_tokens"] for r in generations),
        total_tokens=sum(r["total_tokens"] for r in generations),
        wall_s=elapsed,
        h100_seconds=elapsed,
        resume_wall_s=time.monotonic() - started,
        recovery_triggers=sum(t["plan"]["compute"].get("recovery_triggered", False) for t in trace),
        acceptance_rule_version=label.rule_version
        if label
        else trace[-1]["acceptance"]["rule_version"],
    )
    if judge:
        result.update(
            goal_type=scenario["target"][0],
            outcome_label=trace[-1]["acceptance"]["label"],
            judge_protocol_sha256=judge.protocol_hash,
        )
    atomic_json(snapshot_path, result)
    return result
