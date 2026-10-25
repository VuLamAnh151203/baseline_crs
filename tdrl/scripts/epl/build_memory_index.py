#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tdrl.baselines.epl.config import EPLConfig
from tdrl.baselines.epl.memory import NativeMiniLMEmbedding, embed_experiences
from tdrl.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description="Build native MiniLM embeddings for EPL train memory")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("TopDial", "DuRecDial2.0"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--retrieval-encoder", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    args = parser.parse_args()
    config = EPLConfig.from_path(args.config)
    if config.dataset != args.dataset:
        raise ValueError("EPL memory index dataset/config mismatch")
    memory = args.run_root / "memory" / args.dataset / "experiences.jsonl"
    vectors = args.run_root / "memory" / args.dataset / "vectors.pt"
    backend = NativeMiniLMEmbedding(
        args.encoder,
        args.retrieval_encoder,
        args.model_lock,
        args.run_root / "memory" / args.dataset / "embedding-journal",
    )
    payload = embed_experiences(
        memory,
        vectors,
        backend,
        config,
        f"epl/{args.dataset}/{file_hash(memory)}",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset": args.dataset,
                "experience_count": len(payload["experience_ids"]),
                "memory_sha256": file_hash(memory),
                "vector_sha256": file_hash(vectors),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
