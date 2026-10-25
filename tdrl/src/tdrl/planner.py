import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace

from .acceptance import normalize
from .autonomy import REFUSAL_RESPONSE, explicit_refusal
from .contracts import Decoding, Message, Plan
from .provenance import digest
from .strategies import ACTION_INSTRUCTIONS

INSTRUCTION = "Represent the observed dialogue state for directed reachability retrieval."
ASSISTANT_POLICY = (
    "You are a helpful conversational recommender. Use only the supplied conversation and "
    "permitted guidance. Training guidance specifies a conversational strategy, not facts about "
    "this user. Use the current target and permitted knowledge; do not copy reference people, "
    "items, preferences or factual claims. Suggest the target only when appropriate. Be natural and honest. Respect explicit refusal; do not repeat "
    "persuasion after refusal. Keep replies to one or two short sentences. "
    "Return only your final conversational response."
)


def derived_seed(seed, seed_key, purpose, candidate=0, depth=0):
    return int(digest([seed, seed_key, purpose, candidate, depth])[:8], 16) % (2**31)


def target_fact_guidance(visible_context):
    context = json.loads(visible_context)
    if set(context) != {"target", "profile", "knowledge"}:
        raise ValueError("grounding requires the shared visible context only")
    target, knowledge = context["target"], context["knowledge"]
    if (
        not isinstance(target, list)
        or len(target) != 2
        or not all(isinstance(x, str) and x.strip() for x in target)
    ):
        raise ValueError("invalid grounding target")
    if not isinstance(knowledge, list) or any(
        not isinstance(row, list) or len(row) != 3 or not isinstance(row[0], str)
        for row in knowledge
    ):
        raise ValueError("invalid grounding knowledge triples")
    facts = [row for row in knowledge if normalize(row[0]) == normalize(target[1])]
    return (
        "\nCurrent target facts (a subset of the already supplied knowledge): "
        + json.dumps({"target": target, "facts": facts}, ensure_ascii=False, sort_keys=True)
        + "\nKeep each fact attached to its named subject. Do not transfer another item's plot, "
        "rating, ingredients or attributes to this target, or use training examples as facts. "
        "Answer the user's latest question directly from relevant permitted knowledge; if the "
        "requested target detail is absent, say it is unavailable instead of inventing it. "
        "Respect refusal and leave the choice to the user."
    )


@dataclass(frozen=True)
class PlannerConfig:
    h_cover: float
    candidates: int = 3
    lookahead: int = 2
    routing: str = "conditional"
    random_probability: float = 0.5
    generation_call_cap: int | None = None
    revision: int = 1
    reference_selection: str = "entry_plus_path"
    preserve_retrieval_candidate: bool = False
    semantic_rollout_terminals: bool = False
    use_observed_guidance: bool = False
    ground_target_facts: bool = False

    def retrieval_control(self):
        return replace(self, routing="retrieval_only", semantic_rollout_terminals=False)

    def __post_init__(self):
        if type(self.ground_target_facts) is not bool or (
            self.ground_target_facts and self.revision != 2
        ):
            raise ValueError("target grounding requires planner revision 2")
        if type(self.use_observed_guidance) is not bool or (
            self.use_observed_guidance and self.revision != 2
        ):
            raise ValueError("observed guidance requires planner revision 2")
        if self.revision not in {1, 2}:
            raise ValueError("unknown planner revision")
        if self.reference_selection not in {"entry_plus_path", "nearest_entry"}:
            raise ValueError("unknown reference selection")
        if not isinstance(self.preserve_retrieval_candidate, bool) or (
            self.preserve_retrieval_candidate and self.revision != 2
        ):
            raise ValueError("retrieval candidate requires planner revision 2")
        if not math.isfinite(self.h_cover) or self.h_cover < 0:
            raise ValueError("invalid H_cover")
        if type(self.semantic_rollout_terminals) is not bool or (
            self.semantic_rollout_terminals and self.revision != 2
        ):
            raise ValueError("semantic rollout terminals require planner revision 2")
        if self.candidates < 1 or self.lookahead < 1:
            raise ValueError("invalid recovery budget")
        if self.routing not in {"conditional", "retrieval_only", "always", "random"}:
            raise ValueError("unknown routing")
        if not 0 <= self.random_probability <= 1:
            raise ValueError("invalid random routing probability")
        if self.generation_call_cap is not None and self.generation_call_cap < 1:
            raise ValueError("must reserve at least one final response call")


class ConditionalPlanner:
    def __init__(
        self,
        generation,
        embedding,
        graph,
        cfg: PlannerConfig,
        decoding=None,
        simulator_decoding=None,
        simulator_prompt="",
        visible_context="",
        rollout_judge=None,
        rollout_scenario=None,
    ):
        self.generation, self.embedding, self.graph = generation, embedding, graph
        self.cfg, self.decoding = cfg, decoding or Decoding()
        self.simulator_decoding = simulator_decoding or Decoding()
        self.simulator_prompt, self.visible_context = simulator_prompt, visible_context
        self.target_grounding = (
            target_fact_guidance(visible_context) if cfg.ground_target_facts else ""
        )
        self.rollout_judge, self.rollout_scenario = rollout_judge, rollout_scenario
        if cfg.semantic_rollout_terminals and (rollout_judge is None or rollout_scenario is None):
            raise ValueError(
                "semantic rollout terminals require the frozen local judge and scenario"
            )

    def _generate(self, state, guidance, seed, request_id):
        system = (
            ASSISTANT_POLICY
            + "\n"
            + self.visible_context
            + "\nGuidance: "
            + guidance
            + self.target_grounding
        )
        return self.generation.generate(
            [Message("system", system), *state.history],
            "assistant",
            self.decoding,
            seed,
            request_id,
        )

    def act(self, state, target_nodes: list[str], seed: int, request_id: str) -> Plan:
        import random

        start = time.monotonic()
        seed_key = getattr(self, "seed_context", request_id)
        gen_start, emb_start = len(self.generation.requests), len(self.embedding.requests)
        replay_ids = []

        def embed(s, rid, ids=None):
            (replay_ids if ids is None else ids).append(rid)
            return self.embedding.embed([s.serialize()], INSTRUCTION, rid, rid).vectors[0]

        live = embed(state, request_id + "/live")
        if not getattr(self, "semantic_judging", False) and explicit_refusal(state):
            return Plan(
                REFUSAL_RESPONSE,
                False,
                None,
                {},
                [],
                None,
                [],
                {
                    "generation_requests": 0,
                    "embedding_requests": 1,
                    "recovery_triggered": False,
                    "refusal_guard": True,
                    "planner_latency_s": time.monotonic() - start,
                },
                replay_ids,
            )
        retrieved = self.graph.retrieve(live, target_nodes) if self.graph is not None else []
        covered = bool(retrieved and retrieved[0][0] <= self.cfg.h_cover)
        distances = {node: d for d, node, _ in retrieved}
        ranked = retrieved
        if (
            self.cfg.revision == 2
            and self.cfg.reference_selection == "entry_plus_path"
            and retrieved
        ):
            supported = [r for r in retrieved if r[0] <= self.cfg.h_cover]
            ranked = sorted(
                supported or retrieved,
                key=lambda r: (r[0] + self.graph.remaining_cost(r[2]), r[0], r[1]),
            )
        reference_id = ranked[0][1] if ranked else None
        path = ranked[0][2] if ranked else []
        recovery = not covered
        if self.cfg.routing == "retrieval_only":
            recovery = False
        elif self.cfg.routing == "always":
            recovery = True
        elif self.cfg.routing == "random":
            recovery = (
                random.Random(derived_seed(seed, seed_key, "routing")).random()
                < self.cfg.random_probability
            )
        guidance = self.graph.references[reference_id].guidance if reference_id else ""
        requested_recovery = recovery
        budget_exhausted = (
            recovery
            and self.cfg.generation_call_cap is not None
            and self.cfg.candidates
            * (3 if self.cfg.semantic_rollout_terminals else 2)
            * self.cfg.lookahead
            > self.cfg.generation_call_cap
        )
        if budget_exhausted:
            recovery = False
        if path and self.cfg.revision == 1:
            guidance = "Use the next move from this directed offline guidance path:\n" + "\n".join(
                self.graph.references[node].guidance for node in path[:3]
            )
        if self.cfg.revision == 2 and reference_id:
            action = self.graph.references[reference_id].action
            if action not in ACTION_INSTRUCTIONS:
                raise ValueError("actionable map requires a declared training strategy")
            guidance = ACTION_INSTRUCTIONS[action]
            if self.cfg.use_observed_guidance:
                guidance += (
                    "\nTraining next-move example, used only for conversational form and intent. "
                    "Adapt it to the current conversation; never copy its names, preferences, "
                    "facts or claims. Use only current permitted knowledge:\n"
                    + self.graph.references[reference_id].guidance
                )
        candidate_actions = list(
            dict.fromkeys(
                [
                    self.graph.references[reference_id].action
                    if reference_id and self.cfg.revision == 2
                    else "clarify",
                    "explain",
                    "recommend",
                    "clarify",
                ]
            )
        )
        scores, texts, terminal_traces = [], [], []
        if recovery:
            if not self.simulator_prompt:
                raise ValueError("recovery requires the frozen user simulator protocol")
            workers = int(os.environ.get("TDRL_CANDIDATE_WORKERS", "1"))
            if workers not in {1, 3}:
                raise ValueError("candidate execution requires one or three workers")

            def branch_rollout(candidate):
                ids = []
                judgments = []
                rid = f"{request_id}/candidate/{candidate}"
                ids.append(rid)
                response = self._generate(
                    state,
                    (
                        guidance
                        if candidate == 0 and self.cfg.preserve_retrieval_candidate
                        else ACTION_INSTRUCTIONS[
                            candidate_actions[candidate % len(candidate_actions)]
                        ]
                        if self.cfg.revision == 2
                        else "Explore a suitable next move."
                    ),
                    derived_seed(seed, seed_key, "final")
                    if candidate == 0 and self.cfg.preserve_retrieval_candidate
                    else derived_seed(seed, seed_key, "candidate", candidate),
                    rid,
                ).text
                branch = state.append(Message("assistant", response))
                for depth in range(self.cfg.lookahead):
                    rid = f"{request_id}/rollout/{candidate}/{depth}/user"
                    ids.append(rid)
                    # Flip conversational roles: the user simulator is the responding model.
                    flipped = [
                        Message("assistant" if m.role == "user" else "user", m.content)
                        for m in branch.history
                    ]
                    user = self.generation.generate(
                        [Message("system", self.simulator_prompt), *flipped],
                        "rollout_user",
                        self.simulator_decoding,
                        derived_seed(seed, seed_key, "rollout_user", candidate, depth),
                        rid,
                    )
                    branch = branch.append(Message("user", user.text))
                    if self.cfg.semantic_rollout_terminals:
                        rid = f"{request_id}/rollout/{candidate}/{depth}/judge"
                        ids.append(rid)
                        verdict = self.rollout_judge.assess(
                            branch,
                            {
                                **self.rollout_scenario,
                                "latest_user_reply_finish_reason": user.finish_reason,
                            },
                            derived_seed(seed, seed_key, "rollout_judge", candidate, depth),
                            rid,
                        )
                        judgments.append(asdict(verdict))
                        if verdict.accepted or verdict.refused or verdict.pressured:
                            break
                    elif explicit_refusal(branch):
                        break
                    if depth + 1 < self.cfg.lookahead:
                        rid = f"{request_id}/rollout/{candidate}/{depth}/assistant"
                        ids.append(rid)
                        reply = self._generate(
                            branch,
                            "Continue naturally; respect refusal.",
                            derived_seed(seed, seed_key, "rollout_assistant", candidate, depth),
                            rid,
                        )
                        branch = branch.append(Message("assistant", reply.text))
                leaf = embed(branch, f"{request_id}/leaf/{candidate}", ids)
                return response, leaf, ids, judgments

            if workers == 1:
                branches = [branch_rollout(i) for i in range(self.cfg.candidates)]
            else:
                # map assembles by candidate index; shutdown drains all requests on error.
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="candidate"
                ) as pool:
                    branches = list(pool.map(branch_rollout, range(self.cfg.candidates)))
            # Score frozen heads and access the graph cache on the controller thread.
            for response, leaf, ids, judgments in branches:
                texts.append(response)
                replay_ids.extend(ids)
                terminal_traces.append(judgments)
                if judgments and judgments[-1]["accepted"]:
                    scores.append(0.0)
                    continue
                if judgments and (judgments[-1]["refused"] or judgments[-1]["pressured"]):
                    scores.append(math.inf)
                    continue
                leaf_refs = (
                    self.graph.retrieve(leaf, target_nodes) if self.graph is not None else []
                )
                scores.append(
                    min(d + self.graph.remaining_cost(p) for d, _, p in leaf_refs)
                    if leaf_refs and self.cfg.revision == 2
                    else leaf_refs[0][0]
                    if leaf_refs
                    else math.inf
                )
            finite = [i for i, score in enumerate(scores) if math.isfinite(score)]
            chosen = (
                min(
                    finite,
                    key=lambda i: (
                        not (terminal_traces[i] and terminal_traces[i][-1]["accepted"])
                        if self.cfg.semantic_rollout_terminals
                        else False,
                        scores[i],
                        i,
                    ),
                )
                if finite
                else 0
            )
            response = texts[chosen]
        else:
            chosen = None
            rid = request_id + "/final"
            replay_ids.append(rid)
            response = self._generate(
                state, guidance, derived_seed(seed, seed_key, "final"), rid
            ).text
        generation = [
            r
            for r in self.generation.requests[gen_start:]
            if r.request_id.startswith(request_id + "/")
        ]
        embeddings = [
            r
            for r in self.embedding.requests[emb_start:]
            if r.request_id.startswith(request_id + "/")
        ]
        compute = {
            "generation_requests": len(generation),
            "embedding_requests": len(embeddings),
            "input_tokens": sum(x.input_tokens for x in generation),
            "output_tokens": sum(x.output_tokens for x in generation),
            "total_tokens": sum(x.total_tokens for x in generation),
            "embedding_texts": sum(x.batch_size for x in embeddings),
            "embedding_tokens": sum(x.input_tokens or 0 for x in embeddings),
            "planner_latency_s": time.monotonic() - start,
            "recovery_triggered": recovery,
            "recovery_requested": requested_recovery,
            "recovery_budget_exhausted": budget_exhausted,
            "prompt_sha256": digest(ASSISTANT_POLICY),
            "config": asdict(self.cfg),
            "rollout_judgments": terminal_traces,
            "rollout_judge_protocol_sha256": self.rollout_judge.protocol_hash
            if self.cfg.semantic_rollout_terminals
            else None,
            "goal_conditioning": self.graph.provenance.get("goal_conditioning")
            if self.graph
            else "no_map",
            "reference_remaining_cost": self.graph.remaining_cost(path)
            if self.graph and path
            else None,
            "leaf_scoring": "directed_entry_plus_path_cost"
            if self.cfg.revision == 2
            else "directed_entry_only",
        }
        # Null means no supported leaf; preserve it without serializing infinity.
        return Plan(
            response,
            covered,
            reference_id,
            distances,
            path,
            chosen,
            [s if math.isfinite(s) else None for s in scores],
            compute,
            replay_ids,
        )
