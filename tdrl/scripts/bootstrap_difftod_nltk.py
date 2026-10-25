#!/usr/bin/env python3
"""Pinned NLTK English tokenizer bootstrap only; no inference or credentials."""

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("configs/difftod-nltk.lock.json"))
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    if lock.get("schema_version") != 1 or lock["nltk_version"] != "3.9.1":
        raise ValueError("unknown tokenizer resource pin")
    url = (
        "https://raw.githubusercontent.com/nltk/nltk_data/"
        + lock["revision"]
        + "/"
        + lock["archive"]
    )
    data = urllib.request.urlopen(url, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != lock["archive_sha256"]:
        raise ValueError("tokenizer archive hash mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for relative, expected in lock["files"].items():
            if (
                not relative.startswith("tokenizers/punkt_tab/english/")
                or ".." in Path(relative).parts
            ):
                raise ValueError("unsafe/unexpected tokenizer resource")
            content = archive.read(relative.removeprefix("tokenizers/"))
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("tokenizer resource hash mismatch")
            path = args.output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != content:
                raise ValueError("existing tokenizer resource mismatch; preserve it")
            path.write_bytes(content)
    print(
        json.dumps(
            {
                "status": "complete",
                "inference_performed": False,
                "files_verified": len(lock["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
