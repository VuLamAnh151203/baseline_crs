"""Bootstrap import/CLI check only: does not initialize models or execute inference."""

import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

import vllm.platforms
from vllm.platforms.cpu import CpuPlatform


def main():
    # Login nodes have no CUDA device. This sets parser defaults only; no model is created.
    # The Slurm server subprocess does not execute this bootstrap-only script.
    vllm.platforms.current_platform = CpuPlatform()

    from vllm.entrypoints.launchers.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from tdrl.backend import MODEL_IDS
    from tdrl.runtime import runtime_path, server_command

    ninja = shutil.which("ninja", path=runtime_path())
    if ninja is None:
        raise RuntimeError("Pinned ninja executable missing from the server runtime PATH")
    version = subprocess.check_output([ninja, "--version"], text=True).strip()
    if importlib.metadata.version("ninja") != "1.13.2" or version.split(".git.", 1)[0] != "1.13.2":
        raise RuntimeError(f"Unexpected ninja build tool version: {version}")
    print(json.dumps({"tool": "ninja", "path": ninja, "version": version}))

    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "configs/runtime.json").read_text())
    for kind in ("generation", "embedding"):
        command = server_command(kind, "bootstrap-path-not-loaded", MODEL_IDS[kind], settings[kind])
        parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(command[3:])
        validate_parsed_serve_args(parsed)
        print(
            json.dumps(
                {"kind": kind, "status": "arguments_validated_no_inference", "command": command}
            )
        )


if __name__ == "__main__":
    main()
