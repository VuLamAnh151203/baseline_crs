#!/usr/bin/env python3
"""Anonymous pinned DiffTOD policy/tokenizer bootstrap; never run inference."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def bootstrap(lock_path, output):
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != 1:
        raise ValueError("unknown DiffTOD model lock version")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "bootstrap-status.json"
    contract = {"model_lock_sha256": sha(lock_path), "lock": lock}
    status = {
        "schema_version": 1,
        "status": "partial",
        "contract": contract,
        "inference_performed": False,
        "verified_files": {},
    }
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("schema_version") != 1 or previous["contract"] != contract:
            raise ValueError("bootstrap provenance mismatch; retain old files")
        status = previous
    status["status"] = "partial"
    status["started_at"] = time.time()

    def record():
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    record()
    try:
        # Small tokenizer downloads precede the 1.7GB initialization.
        for model in [lock["tokenizer"], lock["policy"]]:
            revision = model["revision"]
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("bootstrap requires immutable model revisions")
            directory = output / model["model_id"].split("/")[-1] / revision
            directory.mkdir(parents=True, exist_ok=True)
            for filename, expected in model["files"].items():
                if Path(filename).name != filename or len(expected) != 64:
                    raise ValueError("invalid bootstrap path/hash")
                destination = directory / filename
                partial = destination.with_name(filename + ".partial")
                if destination.exists():
                    if sha(destination) != expected:
                        raise ValueError("existing bootstrap checksum mismatch: " + filename)
                else:
                    if not partial.exists() or sha(partial) != expected:
                        url = (
                            "https://huggingface.co/"
                            + model["model_id"]
                            + "/resolve/"
                            + revision
                            + "/"
                            + filename
                        )
                        with (directory / (filename + ".transfer.log")).open("a") as log:
                            subprocess.run(
                                [
                                    "curl",
                                    "--disable",
                                    "--location",
                                    "--fail",
                                    "--connect-timeout",
                                    "20",
                                    "--max-time",
                                    "1800",
                                    "--retry",
                                    "3",
                                    "--retry-delay",
                                    "2",
                                    "--continue-at",
                                    "-",
                                    url,
                                    "-o",
                                    str(partial),
                                ],
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )
                    if sha(partial) != expected:
                        raise ValueError("download checksum mismatch: " + filename)
                    partial.replace(destination)
                key = str(destination.relative_to(output))
                status["verified_files"][key] = expected
                record()
                print("Verified " + key, flush=True)
        status.update(status="complete", ended_at=time.time())
        status.pop("error", None)
        record()
    except Exception as error:
        status.update(status="partial", ended_at=time.time(), error=str(error))
        record()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("configs/difftod-policy-model.lock.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bootstrap(args.lock, args.output)


if __name__ == "__main__":
    main()
