#!/usr/bin/env python3
"""Tokenize audited TopDial train/dev for DiffTOD; no inference or test access."""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tdrl.diffusion_data import SERIALIZATION_VERSION, load_diffusion_split, pack_dialogues
from tdrl.provenance import atomic_json, digest, file_hash


def main(split_loader=load_diffusion_split, serialization=SERIALIZATION_VERSION):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    lock_path = root / "configs/difftod-policy-model.lock.json"
    lock = json.loads(lock_path.read_text())
    if (
        lock["schema_version"] != 1
        or file_hash(args.tokenizer) != lock["tokenizer"]["files"]["tokenizer.json"]
    ):
        raise ValueError("unverified diffusion tokenizer")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos != 50256 or tokenizer.get_vocab_size() != 50257:
        raise ValueError("native GPT2 tokenizer contract mismatch")
    manifest_path = root / "data/processed/data-manifest.json"
    data_manifest = json.loads(manifest_path.read_text())
    rows = {split: split_loader(root, split, data_manifest) for split in ["train", "dev"]}
    if {r["id"] for r in rows["train"]} & {r["id"] for r in rows["dev"]}:
        raise ValueError("diffusion train/development overlap")
    contract = {
        "schema_version": 1,
        "serialization": serialization,
        "data_manifest_sha256": file_hash(manifest_path),
        "model_lock_sha256": file_hash(lock_path),
        "tokenizer_sha256": file_hash(args.tokenizer),
        "context_length": 1024,
        "packing_group_size": 1000,
        "eos_token_id": eos,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "manifest.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if (
            previous.get("schema_version") != 1
            or previous["contract"] != contract
            or previous["run_id"] != digest(contract)
            or previous["status"] not in {"partial", "complete"}
        ):
            raise ValueError("diffusion preparation contract mismatch; preserve old output")
        for name, sha in previous["outputs"].items():
            if file_hash(args.output / name) != sha:
                raise ValueError("diffusion prepared output checksum mismatch")
        if previous["status"] == "complete":
            print(json.dumps({"status": "reused", "manifest_sha256": file_hash(metadata_path)}))
            return
        if {p.name for p in args.output.iterdir()} != {"manifest.json", *previous["outputs"]}:
            raise ValueError("unmanifested partial diffusion files exist; preserve and audit")
    elif list(args.output.iterdir()):
        raise ValueError("unmanifested diffusion output exists; preserve and audit")
    result = (
        previous
        if metadata_path.exists()
        else {
            "schema_version": 1,
            "status": "partial",
            "contract": contract,
            "run_id": digest(contract),
            "splits": {},
            "outputs": {},
            "inference_performed": False,
        }
    )
    atomic_json(metadata_path, result)
    for split, cases in rows.items():
        if split in result["splits"]:
            continue
        chunks, groups = pack_dialogues(cases, lambda text: tokenizer.encode(text).ids, eos)
        path = args.output / (split + ".npy")
        np.save(path, chunks, allow_pickle=False)
        ids_path = args.output / (split + "-packing.json")
        atomic_json(ids_path, groups)
        result["splits"][split] = {
            "dialogues": len(cases),
            "chunks": len(chunks),
            "dropped_remainder_tokens": sum(g["dropped_remainder_tokens"] for g in groups),
        }
        result["outputs"].update({path.name: file_hash(path), ids_path.name: file_hash(ids_path)})
        atomic_json(metadata_path, result)
    result["status"] = "complete"
    atomic_json(metadata_path, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "splits": result["splits"],
                "manifest_sha256": file_hash(metadata_path),
            }
        )
    )


if __name__ == "__main__":
    main()
