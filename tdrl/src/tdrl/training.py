"""Training-only state banks, offline live-model embeddings, and resumable optimization."""

import json
import random
import time
from dataclasses import asdict

import torch

from .acceptance import classify_user, normalize
from .backend import require_h100_slurm
from .contracts import Message
from .graph import DirectedGraph, Reference
from .learning import CostConfig, ReachabilityLearner
from .planner import INSTRUCTION
from .provenance import atomic_json, digest, file_hash, stop_if_requested
from .reachability import IQE, MRNFixed, assert_invariants
from .state import SERIALIZATION_HASH, DialogueState


def load_prepared(root, dataset, split):
    if split not in {"train", "dev"}:
        raise ValueError("training/selection loaders cannot open official test resources")
    p = root / "data/processed" / dataset / (split + ".jsonl")
    return [json.loads(line) for line in p.read_text().splitlines()]


def state_records(scenarios, dataset, split, terminal_protocol="legacy_v1"):
    if terminal_protocol not in {"legacy_v1", "censored_v2"}:
        raise ValueError("unknown offline terminal protocol")
    records = []
    for scenario in scenarios:
        history = [Message(**m) for m in scenario["messages"]]
        for t, message in enumerate(history):
            if message.role == "assistant" and t > 0 and history[t - 1].role == "user":
                records.append(
                    {
                        "node_id": scenario["id"] + f":{t}",
                        "dialogue_id": scenario["id"],
                        "dataset": dataset,
                        "split": split,
                        "turn": t,
                        "state": DialogueState(tuple(history[:t])).serialize(),
                        "guidance": message.content,
                        "target": scenario["target"],
                        "action": scenario["assistant_actions"][t],
                        "goal_reference": normalize(scenario["target"][1])
                        in normalize(message.content),
                        "terminal": False,
                        "failed_terminal": False,
                    }
                )
        state = DialogueState(tuple(history))
        # An observed trajectory end is not evidence of interactive target acceptance.
        accepted = (
            terminal_protocol == "legacy_v1"
            and history[-1].role == "user"
            and classify_user(state, scenario["target"]).accepted
        )
        records.append(
            {
                "node_id": scenario["id"] + ":terminal",
                "dialogue_id": scenario["id"],
                "dataset": dataset,
                "split": split,
                "turn": len(history),
                "state": state.serialize(),
                "guidance": "",
                "target": scenario["target"],
                "action": None,
                "goal_reference": False,
                "terminal": True,
                "failed_terminal": not accepted,
                "terminal_label_source": terminal_protocol,
            }
        )
    return records


def atomic_torch(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temp)
    temp.replace(path)


def embed_records(records, backend, directory, contract, batch_size=16):
    directory.mkdir(parents=True, exist_ok=True)
    key = digest(contract)
    chunks = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        path = directory / f"chunk-{offset:08d}.pt"
        sidecar = path.with_suffix(".json")
        expected = {
            "contract_sha256": key,
            "node_ids": [r["node_id"] for r in batch],
            "states_sha256": digest([r["state"] for r in batch]),
        }
        if path.exists() or sidecar.exists():
            if not path.exists() or not sidecar.exists():
                raise ValueError("incomplete embedding chunk; retain it and resolve before reuse")
            metadata = json.loads(sidecar.read_text())
            if any(metadata.get(k) != v for k, v in expected.items()) or metadata[
                "sha256"
            ] != file_hash(path):
                raise ValueError("embedding cache provenance mismatch")
        else:
            rid = f"offline/{key}/{offset}"
            result = backend.embed([r["state"] for r in batch], INSTRUCTION, rid, rid)
            atomic_torch(path, result.vectors)
            atomic_json(
                sidecar,
                {
                    **expected,
                    "sha256": file_hash(path),
                    "model": asdict(result.model),
                    "instruction_sha256": result.instruction_sha256,
                    "preprocessing_sha256": result.preprocessing_sha256,
                    "batch_size": result.batch_size,
                    "latency_s": result.latency_s,
                    "input_tokens": result.input_tokens,
                },
            )
        chunks.append(path.name)
    atomic_json(
        directory / "index.json",
        {
            "schema_version": 1,
            "contract": contract,
            "records": records,
            "chunks": chunks,
            "count": len(records),
            "serialization_sha256": SERIALIZATION_HASH,
        },
    )


def load_embeddings(directory):
    index = json.loads((directory / "index.json").read_text())
    if index["schema_version"] != 1 or index["serialization_sha256"] != SERIALIZATION_HASH:
        raise ValueError("embedding-space version mismatch")
    vectors = []
    for name in index["chunks"]:
        path = directory / name
        meta = json.loads(path.with_suffix(".json").read_text())
        if file_hash(path) != meta["sha256"]:
            raise ValueError("embedding chunk checksum failure")
        vectors.append(torch.load(path, map_location="cpu", weights_only=True))
    vectors = torch.cat(vectors)
    if (
        vectors.shape != (len(index["records"]), 4096)
        or not torch.isfinite(vectors).all()
        or not torch.allclose(vectors.norm(dim=-1), torch.ones(len(vectors)), atol=1e-5, rtol=1e-5)
    ):
        raise ValueError("invalid normalized full-size embedding bank")
    return index["records"], vectors


class PairSampler:
    """Positive within-trajectory reachability; terminal/cross-trajectory pairs are censored.

    A failed/censored final state is never a reachable Bellman goal. Negative separation
    does not claim that arbitrary cross-trajectory pairs are universally unreachable.
    """

    def __init__(self, records, seed):
        self.rng = random.Random(seed)
        by_dialogue = {}
        for i, record in enumerate(records):
            by_dialogue.setdefault((record["dataset"], record["dialogue_id"]), []).append(i)
        self.trajectories = [indices for indices in by_dialogue.values() if len(indices) > 1]
        self.records = records
        self.goal_pool = [i for i, r in enumerate(records) if not r["failed_terminal"]]
        if not self.trajectories or not self.goal_pool:
            raise ValueError("no valid transition/goal resources")

    def sample(self, count):
        rows = []
        for _ in range(count):
            trajectory = self.rng.choice(self.trajectories)
            position = self.rng.randrange(len(trajectory) - 1)
            source, successor = trajectory[position : position + 2]
            available = [i for i in trajectory[position:] if not self.records[i]["failed_terminal"]]
            positive = self.rng.random() >= 0.25 and bool(available)
            if positive:
                goal = self.rng.choice(available)
            else:
                for _ in range(100):
                    goal = self.rng.choice(self.goal_pool)
                    if (self.records[goal]["dataset"], self.records[goal]["dialogue_id"]) != (
                        self.records[source]["dataset"],
                        self.records[source]["dialogue_id"],
                    ):
                        break
                else:
                    raise ValueError("no independently censored goal pairs")
                if self.records[trajectory[-1]]["failed_terminal"] and self.rng.random() < 0.5:
                    source = successor = trajectory[-1]
            rows.append((source, successor, goal, source == goal, successor == goal, positive))
        return list(zip(*rows, strict=True))


def train_head(head_type, records, vectors, cfg, directory, provenance):
    require_h100_slurm()
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    head = {"MRNFixed": MRNFixed, "IQE": IQE}[head_type]().to("cuda", torch.bfloat16)
    learner = ReachabilityLearner(
        head,
        CostConfig(**cfg["cost"]),
        lr=cfg["lr"],
        numerical_revision=cfg.get("numerical_revision", 1),
    )
    sampler = PairSampler(records, cfg["seed"])
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["schema_version"] != 1 or saved["provenance"] != provenance:
            raise ValueError("head checkpoint provenance/version mismatch")
        learner.load_state_dict(saved["learner"])
        sampler.rng.setstate(saved["sampler_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        # Optimizer moments must reside with the corresponding CUDA parameters.
        for state in learner.optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor) and name != "step":
                    state[name] = value.to("cuda")
    pool = vectors.to("cuda", torch.bfloat16)
    log = (directory / "loss.jsonl").open("a")
    started = time.monotonic()
    try:
        while learner.steps < cfg["steps"]:
            source, successor, goal, self_goal, next_goal, reachable = sampler.sample(
                cfg["batch_size"]
            )
            loss = learner.step(
                pool[list(source)],
                pool[list(successor)],
                pool[list(goal)],
                torch.tensor(self_goal, device="cuda", dtype=torch.bool),
                torch.tensor(next_goal, device="cuda", dtype=torch.bool),
                torch.tensor(reachable, device="cuda", dtype=torch.bool),
            )
            if learner.steps % 100 == 0:
                log.write(json.dumps({"step": learner.steps, "loss": loss}) + "\n")
                log.flush()
            if learner.steps % 1000 == 0 or learner.steps == cfg["steps"]:
                atomic_torch(
                    checkpoint,
                    {
                        "schema_version": 1,
                        "provenance": provenance,
                        "head_type": head_type,
                        "learner": learner.state_dict(),
                        "sampler_rng": sampler.rng.getstate(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                    },
                )
                atomic_json(
                    directory / "progress.json",
                    {
                        "step": learner.steps,
                        "total_steps": cfg["steps"],
                        "elapsed_s": time.monotonic() - started,
                        "status": "partial",
                    },
                )
                stop_if_requested(directory)
        torch.cuda.synchronize()
        invariants = assert_invariants(
            head, pool[torch.linspace(0, len(pool) - 1, 32).long().to("cuda")], atol=2e-3, rtol=2e-2
        )
        atomic_json(
            directory / "diagnostics.json",
            {
                "invariants": invariants,
                "parameters": sum(p.numel() for p in head.parameters()),
                "checkpoint_sha256": file_hash(checkpoint),
                "negative_pairs": "censored/separation; not universal-unreachable labels",
                "positive_step_unit": "one observed assistant/user exchange",
                "numerical_revision": learner.numerical_revision,
                "forward_parameter_dtype": str(next(head.parameters()).dtype),
                "target_master_dtype": "float32"
                if learner.numerical_revision == 2
                else "same_as_forward",
            },
        )
    finally:
        log.close()
    return head.eval()


def load_head(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["schema_version"] != 1 or saved["learner"].get("numerical_revision", 1) not in {1, 2}:
        raise ValueError("unknown head checkpoint schema/numerical execution")
    head = {"MRNFixed": MRNFixed, "IQE": IQE}[saved["head_type"]]().to("cuda", torch.bfloat16)
    head.load_state_dict(saved["learner"]["online"])
    head.checkpoint_hash = file_hash(path)
    head.encoder_provenance = saved["provenance"]["models"]
    return head.eval()


def make_graph(
    records,
    vectors,
    head,
    h_edge,
    node_budget,
    seed,
    target_type,
    target_topic=None,
    actionable=False,
    reference_scope="goal_type",
    use_observed_guidance=False,
):
    if type(use_observed_guidance) is not bool or (use_observed_guidance and not actionable):
        raise ValueError("observed guidance requires an actionable training graph")
    if reference_scope not in {"goal_type", "exact_target_when_supported"}:
        raise ValueError("unknown training reference scope")
    eligible = [
        i
        for i, r in enumerate(records)
        if r["split"] == "train" and r["guidance"] and r["target"][0] == target_type
    ]
    exact_goals = {
        i
        for i in eligible
        if target_topic is not None
        and records[i]["goal_reference"]
        and normalize(records[i]["target"][1]) == normalize(target_topic)
    }
    if reference_scope == "exact_target_when_supported" and exact_goals:
        eligible = [
            i for i in eligible if normalize(records[i]["target"][1]) == normalize(target_topic)
        ]
    chosen = sorted(
        eligible, key=lambda i: (i not in exact_goals, digest([seed, records[i]["node_id"]]))
    )[:node_budget]
    if not chosen:
        return None, []
    references = []
    for i in chosen:
        r = records[i]
        state = json.loads(r["state"])
        references.append(
            Reference(
                r["node_id"],
                DialogueState(tuple(Message(**m) for m in state["history"])),
                vectors[i],
                r["guidance"]
                if use_observed_guidance
                else "Next assistant strategy observed in training: "
                + r["action"]
                + ". Realize it using the current target, profile and permitted knowledge.",
                r["dialogue_id"],
                action=r["action"] if actionable else None,
            )
        )
    graph = DirectedGraph(
        references,
        head,
        h_edge,
        {
            "checkpoint_sha256": head.checkpoint_hash,
            "target_type": target_type,
            "training_only": True,
            "goal_conditioning": "exact_target" if exact_goals else "goal_type_transfer",
            "target_topic": target_topic,
            "actionable_guidance": actionable,
            "use_observed_guidance": use_observed_guidance,
            "reference_scope": "exact_target"
            if reference_scope == "exact_target_when_supported" and exact_goals
            else "goal_type",
        },
    ).build(block_size=64)
    goals = [
        records[i]["node_id"]
        for i in chosen
        if records[i]["goal_reference"] and (not exact_goals or i in exact_goals)
    ]
    if not goals:
        return None, []
    return graph, goals


def heldout_temporal_diagnostics(head, records, vectors, directory, seed=17, pairs=1024):
    """Labels come from held-out observed chronology, without learned-distance access.

    Reverse chronology is a diagnostic contrast, not a claim of universal unreachability.
    The head is frozen; these development resources never become map nodes or training.
    """
    groups = {}
    for i, record in enumerate(records):
        if record["split"] != "dev":
            raise ValueError("held-out diagnostics require development membership")
        if not record["failed_terminal"]:
            groups.setdefault(record["dialogue_id"], []).append(i)
    trajectories = [g for g in groups.values() if len(g) > 2]
    if not trajectories:
        raise ValueError("insufficient held-out chronological diagnostics")
    rng = random.Random(seed)
    ordered = []
    for _ in range(pairs):
        group = rng.choice(trajectories)
        source, target = sorted(rng.sample(range(len(group)), 2))
        ordered.append((group[source], group[target], target - source))
    parameter = next(head.parameters())
    source = vectors[[p[0] for p in ordered]].to(parameter.device, parameter.dtype)
    target = vectors[[p[1] for p in ordered]].to(parameter.device, parameter.dtype)
    with torch.no_grad():
        forward = head.distance(source, target).float().cpu()
        reverse = head.distance(target, source).float().cpu()
    if not torch.isfinite(forward).all() or not torch.isfinite(reverse).all():
        raise ValueError("nonfinite held-out chronological scoring")
    remaining = torch.tensor([p[2] for p in ordered], dtype=torch.float32)
    correlation = torch.corrcoef(torch.stack([forward, remaining]))[0, 1]
    artifact = {
        "schema_version": 1,
        "status": "complete",
        "split": "dev",
        "pairs": pairs,
        "seed": seed,
        "labels": "observed chronological order and remaining transition count",
        "label_distance_access": False,
        "forward_mean": float(forward.mean()),
        "reverse_mean": float(reverse.mean()),
        "forward_less_than_reverse": float((forward < reverse).float().mean()),
        "remaining_step_pearson": float(correlation) if torch.isfinite(correlation) else None,
        "limitations": "temporal diagnostics, not unreachable labels or general OOD evidence",
        "rows": [
            {
                "source": records[s]["node_id"],
                "target": records[t]["node_id"],
                "remaining_transitions": n,
                "forward": float(f),
                "reverse": float(r),
            }
            for (s, t, n), f, r in zip(ordered, forward, reverse, strict=True)
        ],
    }
    atomic_json(directory / "heldout-temporal.json", artifact)
    return artifact
