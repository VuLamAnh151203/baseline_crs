#!/usr/bin/env python3
"""Freeze DiffTOD loss-validation data without confirmation cases or families."""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tdrl.diffusion_data import load_diffusion_split, pack_dialogues
from tdrl.provenance import atomic_json, digest, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    manifest_path = root / "data/processed/data-manifest.json"
    data_manifest = json.loads(manifest_path.read_text())
    rows = load_diffusion_split(root, "dev", data_manifest)
    membership_path = root / "data/manifests/topdial-development-confirmation64.json"
    frozen = json.loads(membership_path.read_text())
    population = root / "data/processed/TopDial/dev.jsonl"
    if (
        frozen.get("schema_version") != 1
        or frozen["dataset"] != "TopDial"
        or frozen["split"] != "dev"
        or frozen["root_seed"] != 17
        or frozen["population_sha256"] != file_hash(population)
        or frozen["population_count"] != len(rows)
    ):
        raise ValueError("confirmation membership/population contract mismatch")
    ranked = sorted(rows, key=lambda r: digest([17, r["id"]]))
    expected = [r["id"] for r in ranked[80:144]]
    if frozen["scenario_ids"] != expected or len(set(expected)) != 64:
        raise ValueError("confirmation membership/order changed")
    by_id = {r["id"]: r for r in rows}
    for case in expected:
        if digest(by_id[case]) != frozen["scenario_content_sha256"][case]:
            raise ValueError("confirmation normalized content changed")
    families = {case.rsplit("_", 1)[0] for case in expected}
    selected = [r for r in rows if r["id"].rsplit("_", 1)[0] not in families]
    excluded = [r["id"] for r in rows if r["id"].rsplit("_", 1)[0] in families]
    lock = json.loads((root / "configs/difftod-policy-model.lock.json").read_text())
    if (
        lock.get("schema_version") != 1
        or file_hash(args.tokenizer) != lock["tokenizer"]["files"]["tokenizer.json"]
    ):
        raise ValueError("unverified diffusion validation tokenizer")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    contract = {
        "schema_version": 1,
        "scope": "loss-only validation; no confirmation families",
        "confirmation_resource_sha256": file_hash(membership_path),
        "data_manifest_sha256": file_hash(manifest_path),
        "tokenizer_sha256": file_hash(args.tokenizer),
        "selected_ids_sha256": digest([r["id"] for r in selected]),
        "selected_content_sha256": digest(selected),
        "context_length": 1024,
        "packing_group_size": 1000,
        "root_seed": 17,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if (
            previous.get("schema_version") != 1
            or previous["status"] != "complete"
            or previous["contract"] != contract
        ):
            raise ValueError("validation preparation mismatch; preserve old output")
        for name, expected_hash in previous["outputs"].items():
            if file_hash(args.output / name) != expected_hash:
                raise ValueError("validation output checksum mismatch")
        print(json.dumps({"status": "reused", "manifest_sha256": file_hash(path)}))
        return
    if list(args.output.iterdir()):
        raise ValueError("unmanifested validation data; preserve and audit")
    chunks, groups = pack_dialogues(selected, lambda text: tokenizer.encode(text).ids, 50256)
    np.save(args.output / "validation.npy", chunks, allow_pickle=False)
    atomic_json(args.output / "packing.json", groups)
    atomic_json(
        args.output / "membership.json",
        {
            "selected_ids": [r["id"] for r in selected],
            "excluded_confirmation_family_ids": excluded,
            "confirmation_families": sorted(families),
        },
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "contract": contract,
        "run_id": digest(contract),
        "dialogues": len(selected),
        "chunks": len(chunks),
        "excluded_dialogues": len(excluded),
        "outcomes_read": False,
        "dropped_remainder_tokens": sum(g["dropped_remainder_tokens"] for g in groups),
        "outputs": {p.name: file_hash(p) for p in args.output.iterdir()},
    }
    atomic_json(path, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "dialogues": len(selected),
                "chunks": len(chunks),
                "excluded_confirmation_family_dialogues": len(excluded),
                "manifest_sha256": file_hash(path),
            }
        )
    )


if __name__ == "__main__":
    main()
