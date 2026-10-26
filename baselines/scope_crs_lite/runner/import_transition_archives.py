#!/usr/bin/env python3
"""Import split official SCOPE ZIP archives and build an immutable manifest."""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from tdrl.baselines.scope.assets import (
    build_transition_manifest,
    transition_files,
    verify_transition_manifest,
)
from tdrl.provenance import atomic_json, file_hash


def checkpoint_members():
    return {
        f"deterministic/{relative}" for relative in transition_files("unused")
    }


def inspect_archives(archives):
    expected = checkpoint_members()
    owners = {}
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            bad = handle.testzip()
            if bad is not None:
                raise ValueError(f"CRC failure in {archive}: {bad}")
            for info in handle.infolist():
                if info.is_dir():
                    continue
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError(f"unsafe transition archive member: {info.filename}")
                if info.filename not in expected:
                    raise ValueError(f"unexpected transition archive member: {info.filename}")
                if info.filename in owners:
                    raise ValueError(f"duplicate transition archive member: {info.filename}")
                owners[info.filename] = (archive, info)
    missing = expected - set(owners)
    if missing:
        raise ValueError("missing transition archive members: " + ", ".join(sorted(missing)))
    return owners


def extract(owners, staging):
    staging.mkdir(parents=True, exist_ok=True)
    for member, (archive, info) in sorted(owners.items()):
        relative = Path(*PurePosixPath(member).parts[1:])
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with zipfile.ZipFile(archive) as handle, handle.open(info) as source:
            with temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        if temporary.stat().st_size != info.file_size:
            raise ValueError(f"truncated extracted checkpoint: {member}")
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archives = tuple(path.resolve() for path in args.archive)
    if len(archives) != 3 or len(set(archives)) != 3:
        raise ValueError("SCOPE import requires exactly three distinct ZIP archives")
    if any(not path.is_file() for path in archives):
        raise FileNotFoundError("one or more SCOPE transition ZIP archives are absent")
    output = args.output.resolve()
    if output.exists():
        result = verify_transition_manifest(output)
        print(json.dumps({"status": "reused", **result}, indent=2, sort_keys=True))
        return
    owners = inspect_archives(archives)
    staging = output.parent / (output.name + ".import-partial")
    extract(owners, staging)
    manifest = build_transition_manifest(staging)
    os.replace(staging, output)
    verified = verify_transition_manifest(output)
    evidence = {
        "schema_version": 1,
        "status": "complete",
        "archives": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in archives
        ],
        "members": sorted(owners),
        "transition_manifest_sha256": verified["sha256"],
        "transition_files": manifest["files"],
    }
    atomic_json(output.parent / "transition-import-evidence.json", evidence)
    print(json.dumps({"status": "imported", **verified}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
