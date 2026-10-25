#!/usr/bin/env python3
"""Audit completed development guidance; lexical flags are manual-review candidates."""

import argparse
import json
from collections import Counter
from pathlib import Path

from tdrl.provenance import atomic_json, file_hash, validate_manifest


def mentioned(text, phrase):
    return bool(phrase.strip()) and phrase.casefold() in text.casefold()


def audit(rows, scenarios, references, labels):
    actions, first, paths = Counter(), Counter(), Counter()
    flags = []
    for row in rows:
        record = row["record"]
        scenario = scenarios[record["scenario_id"]]
        # The visible context excludes official development continuations.
        allowed = json.dumps(
            {k: scenario[k] for k in ["target", "knowledge", "profile", "personality"]},
            ensure_ascii=False,
        )
        prefix_count = next(
            i + 1 for i, m in enumerate(scenario["messages"]) if m["role"] == "user"
        )
        prefix = list(record["history"][:prefix_count])
        for index, turn in enumerate(record["turns_trace"]):
            plan = turn["plan"]
            ref_id = plan["reference_id"]
            if ref_id is not None:
                ref = references[ref_id]
                if ref["split"] != "train":
                    raise ValueError("non-training guidance")
                action = labels[ref_id]
                actions[action] += 1
                if index == 0:
                    first[action] += 1
                topic = ref["target"][1]
                available = allowed + json.dumps(prefix, ensure_ascii=False)
                if mentioned(plan["response"], topic) and not mentioned(available, topic):
                    flags.append(
                        {
                            "scenario_id": record["scenario_id"],
                            "protocol": record["protocol"],
                            "turn": index,
                            "reference_id": ref_id,
                            "reference_topic": topic,
                            "response": plan["response"],
                        }
                    )
            paths[str(len(plan["path"]))] += 1
            # Append only this executed response and the subsequent observed user turn.
            offset = len(prefix)
            prefix.extend(record["history"][offset : offset + 2])
    return {
        "reference_action_turn_counts": dict(actions),
        "first_reference_action_dialogue_counts": dict(first),
        "path_length_turn_counts": dict(paths),
        "unavailable_reference_topic_flags": flags,
        "limits": "Exact lexical reference-target matches only; flags require manual review and do not establish factual correctness, autonomy, or human quality.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--bank-index", type=Path, required=True)
    p.add_argument("--annotation", type=Path, required=True)
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    m = json.loads((args.evaluation / "manifest.json").read_text())
    validate_manifest(m)
    if (
        m["status"] != "complete"
        or m["stage"] != "TopDial-planner-evaluate"
        or m["provenance"]["dataset"]["split"] != "development"
    ):
        raise ValueError("completed discovery development required; no confirmation/test audit")
    if (
        file_hash(args.evaluation / "records.json") != m["outputs"]["records.json"]
        or file_hash(args.development) != m["provenance"]["dataset"]["processed_sha256"]
    ):
        raise ValueError("development record/population checksum mismatch")
    annotation = json.loads((args.annotation / "manifest.json").read_text())
    validate_manifest(annotation)
    if (
        annotation["status"] != "complete"
        or annotation["stage"] != "TopDial-semantic-strategy-annotate"
    ):
        raise ValueError("completed training annotations required")
    if (
        file_hash(args.bank_index) != annotation["provenance"]["dataset"]["bank_sha256"]
        or file_hash(args.annotation / "labels.json") != annotation["outputs"]["labels.json"]
    ):
        raise ValueError("training guidance checksum mismatch")
    contract = json.loads((args.evaluation / "contract.json").read_text())
    if file_hash(args.evaluation / "contract.json") != m["outputs"]["contract.json"] or contract[
        "strategy_overlay_manifest_sha256"
    ] != file_hash(args.annotation / "manifest.json"):
        raise ValueError("annotation not bound to evaluation")
    scenarios = {x["id"]: x for x in map(json.loads, args.development.read_text().splitlines())}
    bank = json.loads(args.bank_index.read_text())
    references = {r["node_id"]: r for r in bank["records"]}
    labels = {
        r["node_id"]: r["semantic_label"]
        for r in json.loads((args.annotation / "labels.json").read_text())
    }
    result = audit(
        json.loads((args.evaluation / "records.json").read_text()), scenarios, references, labels
    )
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "status": "complete",
            "sources": {
                str(x): file_hash(x)
                for x in [
                    args.evaluation / "manifest.json",
                    args.bank_index,
                    args.annotation / "manifest.json",
                    args.development,
                ]
            },
            **result,
        },
    )


if __name__ == "__main__":
    main()
