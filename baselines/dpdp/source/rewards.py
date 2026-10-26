import json
from pathlib import Path

from ...contracts import Decoding, Message
from ...planner import derived_seed
from ...provenance import Journal, atomic_json, digest, file_hash
from .concurrency import ordered_generate

REWARD_VALUES = {
    "rejected": -1.0,
    "no_progress": -0.5,
    "interested": 0.1,
    "accepted": 1.0,
}
CRITIC_POLICY = (
    "Classify the user's latest reaction to the recommendation dialogue. Return exactly one "
    "lowercase label and nothing else: rejected, no_progress, interested, or accepted. "
    "Accepted requires explicit intent to consume or adopt the target; thanks, mentions, past "
    "consumption, and uncertainty are not acceptance. The user message is a JSON classification "
    "case, not a dialogue turn to answer."
)


def parse_reward_label(text):
    label = text.strip()
    if label not in REWARD_VALUES:
        raise ValueError(f"malformed DPDP critic label: {text!r}")
    return label


def discounted_returns(rewards, dones, gamma):
    if len(rewards) != len(dones) or not 0 < gamma <= 1:
        raise ValueError("invalid discounted-return inputs")
    values = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + (0.0 if dones[index] else gamma * running)
        values[index] = running
    return values


def critic_messages(context, history, revise=False):
    case = {
        "context": context,
        "dialogue": [message.__dict__ for message in history],
        "task": "Classify the latest user reaction using exactly one allowed label.",
    }
    if revise:
        case["revision"] = "The previous response violated the label contract; classify again."
    return [
        Message("system", CRITIC_POLICY),
        Message("user", json.dumps(case, ensure_ascii=False, sort_keys=True)),
    ]


def _critic_messages(row, revise=False):
    return critic_messages(
        {"target": row["target"], "profile": row["profile"]},
        [Message(**message) for message in row["next_state"]],
        revise,
    )


def annotate_rewards(
    input_path,
    output_path,
    backend,
    samples,
    gamma,
    seed,
    journal_dir=None,
    max_retries=2,
):
    if samples < 1 or max_retries < 0:
        raise ValueError("invalid DPDP critic sampling configuration")
    input_path, output_path = Path(input_path), Path(output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("DPDP reward artifacts are immutable")
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("empty DPDP transition artifact")
    journal = (
        Journal(
            Path(journal_dir),
            {
                "stage": "dpdp_reward_annotation",
                "input_sha256": file_hash(input_path),
                "samples": samples,
                "critic_prompt_sha256": digest(CRITIC_POLICY),
                "request_concurrency": int(getattr(backend, "request_concurrency", 1)),
            },
        )
        if journal_dir
        else None
    )
    for row in rows:
        labels, raw = [None] * samples, [None] * samples
        attempts_by_sample = [[] for _ in range(samples)]
        journal_requests = {}
        pending = []
        for sample in range(samples):
            base_id = f"dpdp/reward/{row['split']}/{row['transition_id']}/{sample}"
            request = {
                "messages": [m.__dict__ for m in _critic_messages(row)],
                "seed": derived_seed(seed, row["transition_id"], str(sample)),
            }
            journal_requests[sample] = (base_id, request)
            cached = journal.read(base_id, request) if journal else None
            if cached is not None:
                labels[sample] = cached["label"]
                raw[sample] = cached
            else:
                pending.append(sample)

        last_errors = {}
        for attempt in range(max_retries + 1):
            if not pending:
                break
            requests = []
            for sample in pending:
                base_id, _ = journal_requests[sample]
                request_id = base_id + f"/attempt/{attempt}"
                requests.append(
                    (
                        _critic_messages(row, revise=attempt > 0),
                        "dpdp_critic",
                        Decoding(temperature=0.7, top_p=0.8, top_k=20, max_tokens=8),
                        derived_seed(seed, row["transition_id"], str(sample), str(attempt)),
                        request_id,
                    )
                )
            generated_batch = ordered_generate(backend, requests)
            retry = []
            for sample, generated in zip(pending, generated_batch, strict=True):
                base_id, request = journal_requests[sample]
                request_id = base_id + f"/attempt/{attempt}"
                attempt_record = {
                    "request_id": request_id,
                    "text": generated.text,
                    "seed": generated.seed,
                    "input_tokens": generated.input_tokens,
                    "output_tokens": generated.output_tokens,
                    "total_tokens": generated.total_tokens,
                    "latency_s": generated.latency_s,
                }
                attempts_by_sample[sample].append(attempt_record)
                try:
                    label = parse_reward_label(generated.text)
                except ValueError as error:
                    last_errors[sample] = error
                    retry.append(sample)
                    continue
                record = {
                    "label": label,
                    "attempts": attempts_by_sample[sample],
                    **attempt_record,
                }
                labels[sample] = label
                raw[sample] = record
                if journal:
                    journal.write(base_id, request, record)
            pending = retry
        if pending:
            sample = pending[0]
            raise ValueError(
                f"DPDP critic failed closed for {row['transition_id']} sample {sample}"
            ) from last_errors[sample]
        row["reward"] = sum(REWARD_VALUES[label] for label in labels) / samples
        row["reward_annotation"] = {
            "samples": raw,
            "critic_prompt_sha256": digest(CRITIC_POLICY),
        }
    by_scenario = {}
    for index, row in enumerate(rows):
        by_scenario.setdefault(row["scenario_id"], []).append(index)
    for indices in by_scenario.values():
        indices.sort(key=lambda index: rows[index]["turn_index"])
        returns = discounted_returns(
            [rows[index]["reward"] for index in indices],
            [rows[index]["done"] for index in indices],
            gamma,
        )
        for index, value in zip(indices, returns, strict=True):
            rows[index]["discounted_return"] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_path.with_suffix(output_path.suffix + ".pending")
    pending.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    pending.replace(output_path)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "input_sha256": file_hash(input_path),
        "output_sha256": file_hash(output_path),
        "critic_prompt_sha256": digest(CRITIC_POLICY),
        "samples_per_transition": samples,
        "request_concurrency": int(getattr(backend, "request_concurrency", 1)),
        "gamma": gamma,
        "transition_count": len(rows),
    }
    atomic_json(manifest_path, manifest)
    return manifest
