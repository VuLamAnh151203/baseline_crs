"""DiffTOD semantic-inpainting/consensus policy with shared local Qwen realization."""

import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .backend import EMBEDDING_PREPROCESSING
from .contracts import Decoding, Embeddings, Generation, Message, ModelProvenance, Plan
from .diffusion import sample_analytic, semantic_constraints
from .planner import ASSISTANT_POLICY, INSTRUCTION, derived_seed
from .provenance import atomic_json, digest, file_hash

SEMANTIC_PROMPT = (
    "Write exactly five different brief paraphrases of an appropriate assistant move that "
    "performs the supplied target act for the supplied specific topic. Use only the supplied "
    "target and permitted knowledge. Do not invent user acceptance or preferences. "
    "Each paraphrase must be one line. Return exactly five lines with no indices, bullets, "
    "headings or explanation. Return only final text."
)
PLAN_NOTICE = (
    "The following diffusion plan is hypothetical guidance, not observed user evidence. "
    "Do not copy invented user choices, preferences or facts. Realize the appropriate next "
    "move using the actual conversation and permitted knowledge; respect refusal.\n"
)


def native_plan_text(sample, profile, target):
    """Preserve the native TopDial character-header and EOS-segment extraction."""
    target_text = f"target: {target[0]}, {target[1]}\n"
    segments = sample[len(profile) + len(target_text) :].split("<|endoftext|>")
    if len(segments) == 1 or "user profile:" in segments[1]:
        return segments[0]
    return segments[0] + segments[1]


def native_consensus(plans, nltk_data_path):
    """Native self-BLEU, sentence tokenization, smoothing method 1 and first-index tie."""
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    if importlib.metadata.version("nltk") != "3.9.1":
        raise ValueError("DiffTOD consensus requires pinned NLTK 3.9.1")
    lock_path = Path(__file__).resolve().parents[2] / "configs/difftod-nltk.lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != 1 or lock["nltk_version"] != "3.9.1":
        raise ValueError("unknown consensus tokenizer provenance")
    for relative, expected in lock["files"].items():
        if file_hash(Path(nltk_data_path) / relative) != expected:
            raise ValueError("consensus tokenizer asset checksum mismatch")
    nltk.data.path = [str(nltk_data_path)]
    tokenized = [nltk.word_tokenize(plan) for plan in plans]
    if any(not tokens for tokens in tokenized):
        raise ValueError("empty diffusion trajectory; no consensus fallback")
    scores = [
        sum(
            sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method1)
            for j, reference in enumerate(tokenized)
            if i != j
        )
        for i, hypothesis in enumerate(tokenized)
    ]
    chosen = max(range(len(plans)), key=lambda i: (scores[i], -i))
    return chosen, scores


class DiffTODPlanner:
    def __init__(
        self,
        policy,
        tokenizer,
        generation,
        embedding,
        scenario,
        assistant_decoding,
        guidance_decoding,
        journal_directory,
        policy_provenance,
        nltk_data_path,
        sampling_steps=1024,
        consensus=native_consensus,
    ):
        self.policy, self.tokenizer = policy.eval(), tokenizer
        self.generation, self.embedding = generation, embedding
        self.scenario = scenario
        self.assistant_decoding, self.guidance_decoding = assistant_decoding, guidance_decoding
        self.directory = journal_directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.policy_provenance = policy_provenance
        self.nltk_data_path, self.sampling_steps = nltk_data_path, sampling_steps
        self.consensus = consensus
        self.trace = None

    def _trajectory(self, seed, request_id):
        dialogue = request_id.rsplit("/turn/", 1)[0]
        key = digest(dialogue)
        path = self.directory / (key + ".json")
        pending = self.directory / (key + ".pending.json")
        contract = {
            "schema_version": 1,
            "dialogue_request_id": dialogue,
            "scenario_id": self.scenario["id"],
            "visible_inputs_sha256": digest(
                {k: self.scenario[k] for k in ["target", "profile", "knowledge"]}
            ),
            "policy": self.policy_provenance,
            "seed": seed,
            "sampling_steps": self.sampling_steps,
            "guidance_decoding": asdict(self.guidance_decoding),
            "semantic_prompt_sha256": digest(SEMANTIC_PROMPT),
        }
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("schema_version") != 1 or record["contract"] != contract:
                raise ValueError("diffusion trajectory replay provenance mismatch")
            if record["outputs_sha256"] != digest(record["outputs"]):
                raise ValueError("diffusion trajectory checksum mismatch")
            guide = record["outputs"]["guidance_record"]
            if hasattr(self.generation, "model") and guide["model"] != asdict(
                self.generation.model
            ):
                raise ValueError("cached guidance generation model mismatch")
            if hasattr(self.generation, "consumed_requests"):
                self.generation.consumed_requests.append(
                    Generation(
                        **{
                            **guide,
                            "model": ModelProvenance(**guide["model"]),
                            "decoding": Decoding(**guide["decoding"]),
                        }
                    )
                )
            return record["outputs"]
        if pending.exists():
            raise ValueError("unresolved diffusion planning outcome; preserve before retry")
        atomic_json(pending, {"contract": contract, "started_at": time.time()})
        messages = [
            Message("system", SEMANTIC_PROMPT),
            Message(
                "user",
                json.dumps(
                    {
                        "target_act": self.scenario["target"][0],
                        "target_topic": self.scenario["target"][1],
                        "permitted_knowledge": self.scenario["knowledge"],
                    },
                    ensure_ascii=False,
                ),
            ),
        ]
        guidance_id = dialogue + "/turn/0/diffusion/semantic-guidance"
        result = self.generation.generate(
            messages,
            "planner_guidance",
            self.guidance_decoding,
            derived_seed(seed, dialogue, "semantic_guidance"),
            guidance_id,
        )
        phrases = [line.strip() for line in result.text.splitlines() if line.strip()]
        if len(phrases) != 5 or result.finish_reason == "length":
            raise ValueError("DiffTOD requires exactly five complete semantic paraphrases")
        profile = "user profile: " + "".join(
            f"{k}: {v}\t" for k, v in self.scenario["profile"].items()
        )
        profile_ids = self.tokenizer.encode(profile).ids
        plans, traces = [], []
        started = time.monotonic()
        for index, phrase in enumerate(phrases):
            guidance = self.tokenizer.encode("system: " + phrase).ids
            constraints = semantic_constraints(profile_ids, guidance)
            branch_seed = derived_seed(seed, dialogue, "diffusion_trajectory", index)
            tokens = sample_analytic(
                self.policy,
                constraints,
                self.sampling_steps,
                1,
                "cuda",
                torch.Generator(device="cuda").manual_seed(branch_seed),
            )
            sample = self.tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False)
            plan = native_plan_text(sample, profile, self.scenario["target"])
            plans.append(plan)
            traces.append(
                {
                    "index": index,
                    "seed": branch_seed,
                    "token_ids": tokens[0].tolist(),
                    "constraint_sha256": digest(
                        {
                            "positions": constraints.positions.tolist(),
                            "tokens": constraints.tokens.tolist(),
                        }
                    ),
                }
            )
        chosen, scores = self.consensus(plans, self.nltk_data_path)
        outputs = {
            "plans": plans,
            "chosen": chosen,
            "consensus_scores": scores,
            "paraphrases": phrases,
            "branches": traces,
            "policy_forwards": 5 * (self.sampling_steps + 1),
            "policy_sampling_wall_s": time.monotonic() - started,
            "guidance_request_id": guidance_id,
            "guidance_record": asdict(result),
        }
        atomic_json(
            path,
            {
                "schema_version": 1,
                "contract": contract,
                "outputs": outputs,
                "outputs_sha256": digest(outputs),
            },
        )
        pending.unlink()
        return outputs

    def _live_embedding(self, state, request_id):
        path = self.directory / ("live-" + digest(request_id) + ".json")
        pending = path.with_suffix(".pending.json")
        contract = {
            "schema_version": 1,
            "texts": [state.serialize()],
            "instruction_sha256": digest(INSTRUCTION),
            "preprocessing_sha256": digest(EMBEDDING_PREPROCESSING),
            "model": asdict(self.embedding.model) if hasattr(self.embedding, "model") else None,
            "request_id": request_id,
            "adapter_sha256": file_hash(Path(__file__)),
        }
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("schema_version") != 1 or record["contract"] != contract:
                raise ValueError("completed live embedding provenance mismatch")
            output = record["outputs"]
            if digest(output) != record["outputs_sha256"]:
                raise ValueError("completed live embedding checksum mismatch")
            vectors = torch.tensor(output["vectors"], dtype=torch.float32)
            metadata = output["metadata"]
            if (
                vectors.shape != (1, 4096)
                or not torch.isfinite(vectors).all()
                or not torch.allclose(vectors.norm(dim=-1), torch.ones(1), atol=1e-6, rtol=0)
                or metadata["instruction_sha256"] != contract["instruction_sha256"]
                or metadata["preprocessing_sha256"] != contract["preprocessing_sha256"]
                or metadata["request_id"] != request_id
                or metadata["batch_id"] != request_id
                or metadata["batch_size"] != 1
                or (contract["model"] is not None and metadata["model"] != contract["model"])
            ):
                raise ValueError("completed live embedding shape/order/provenance invalid")
            if pending.exists():
                if json.loads(pending.read_text())["contract"] != contract:
                    raise ValueError("live embedding pending/completed contract mismatch")
                pending.unlink()
            return Embeddings(
                vectors=vectors, **{**metadata, "model": ModelProvenance(**metadata["model"])}
            )
        if pending.exists():
            raise ValueError("unknown live embedding outcome; preserve before retry")
        atomic_json(pending, {"contract": contract, "started_at": time.time()})
        result = self.embedding.embed(contract["texts"], INSTRUCTION, request_id, request_id)
        metadata = {k: v for k, v in asdict(result).items() if k != "vectors"}
        output = {"vectors": result.vectors.tolist(), "metadata": metadata}
        atomic_json(
            path,
            {
                "schema_version": 1,
                "contract": contract,
                "outputs": output,
                "outputs_sha256": digest(output),
            },
        )
        pending.unlink()
        return self._live_embedding(state, request_id)

    def act(self, state, target_nodes, seed, request_id):
        from .evaluation import visible_context

        started = time.monotonic()
        live_id = request_id + "/live"
        self._live_embedding(state, live_id)
        if self.trace is None:
            self.trace = self._trajectory(seed, request_id)
        final_id = request_id + "/final"
        reply = self.generation.generate(
            [
                Message(
                    "system",
                    ASSISTANT_POLICY
                    + "\n"
                    + visible_context(self.scenario)
                    + "\nGuidance: "
                    + PLAN_NOTICE
                    + self.trace["plans"][self.trace["chosen"]],
                ),
                *state.history,
            ],
            "assistant",
            self.assistant_decoding,
            derived_seed(seed, self.seed_context, "final"),
            final_id,
        )
        return Plan(
            reply.text,
            False,
            None,
            {},
            [],
            self.trace["chosen"],
            [],
            {
                "planner_latency_s": time.monotonic() - started,
                "recovery_triggered": False,
                "diffusion": self.trace,
                "policy_provenance": self.policy_provenance,
                "scope": "DiffTOD same-backbone adaptation; completion requires fitting/fidelity gates",
            },
            [live_id, self.trace["guidance_request_id"], final_id],
        )
