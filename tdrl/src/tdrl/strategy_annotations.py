"""Training-only semantic strategy overlay; never modify latent states or source labels."""

import json
from pathlib import Path

from .provenance import file_hash, validate_manifest
from .strategies import ACTION_INSTRUCTIONS


def strategy_payload(record):
    if record["split"] != "train" or not record["guidance"]:
        raise ValueError("semantic strategy annotation requires an observed training move")
    state = json.loads(record["state"])
    return {
        "target": record["target"],
        "preceding_dialogue": state["history"],
        "observed_assistant_move": record["guidance"],
    }


def apply_strategy_overlay(
    records,
    directory: Path,
    bank_sha256,
    exclude_farewell=False,
    exclude_non_target_recommendations=False,
):
    if type(exclude_non_target_recommendations) is not bool:
        raise ValueError("recommendation alignment exclusion must be explicitly boolean")
    if type(exclude_farewell) is not bool:
        raise ValueError("closing-guidance exclusion must be explicitly boolean")
    manifest = json.loads((directory / "manifest.json").read_text())
    validate_manifest(manifest)
    if (
        manifest["status"] != "complete"
        or manifest["stage"] != "TopDial-semantic-strategy-annotate"
    ):
        raise ValueError("complete semantic training annotation required")
    if manifest["provenance"]["dataset"]["bank_sha256"] != bank_sha256:
        raise ValueError("semantic overlay training bank mismatch")
    path = directory / "labels.json"
    if file_hash(path) != manifest["outputs"].get("labels.json"):
        raise ValueError("semantic strategy label checksum mismatch")
    labels = json.loads(path.read_text())
    eligible = {r["node_id"]: r for r in records if r["split"] == "train" and r["guidance"]}
    by_node = {}
    for row in labels:
        node = row["node_id"]
        if node in by_node or node not in eligible or row["weak_label"] != eligible[node]["action"]:
            raise ValueError("duplicate/unexpected strategy node or changed source label")
        if row["semantic_label"] not in {*ACTION_INSTRUCTIONS, "ambiguous"}:
            raise ValueError("unknown semantic strategy; no fallback")
        by_node[node] = row["semantic_label"]
    if by_node.keys() != eligible.keys():
        raise ValueError("partial semantic strategy overlay")
    updated = []
    for record in records:
        copy = dict(record)
        if record["node_id"] in by_node:
            copy["action"] = by_node[record["node_id"]]
            misaligned = (
                exclude_non_target_recommendations
                and copy["action"] == "recommend"
                and not record["goal_reference"]
            )
            if (
                copy["action"] == "ambiguous"
                or (exclude_farewell and copy["action"] == "farewell")
                or misaligned
            ):
                copy.update(action=None, guidance="")
        updated.append(copy)
    return updated
