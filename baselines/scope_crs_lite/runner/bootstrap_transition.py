#!/usr/bin/env python3
"""Download and hash the public official SCOPE transition ensemble."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tdrl.baselines.scope.assets import (
    TRANSITION_FOLDER_ID,
    build_transition_manifest,
    verify_transition_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        result = verify_transition_manifest(output)
        print(json.dumps({"status": "reused", **result}, indent=2, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (output.name + ".download-partial")
    staging.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            f"https://drive.google.com/drive/folders/{TRANSITION_FOLDER_ID}",
            "-O",
            str(staging),
        ],
        check=True,
    )
    downloaded = staging
    nested = staging / "deterministic"
    if not (staging / "seed_0_batch_2048").is_dir() and nested.is_dir():
        downloaded = nested
    build_transition_manifest(downloaded)
    os.replace(downloaded, output)
    result = verify_transition_manifest(output)
    print(json.dumps({"status": "downloaded", **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
