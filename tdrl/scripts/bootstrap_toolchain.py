"""Install a hash-pinned project-local CUDA compiler; compilation only, no inference."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lock_path = root / "configs/cuda-toolchain.lock.json"
lock = json.loads(lock_path.read_text())
if lock["schema_version"] != 1:
    raise ValueError("unknown toolchain lock schema")
base = root / "toolchains"
target = base / ("cuda-" + lock["version"])
marker = target / "manifest.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if marker.exists():
    manifest = json.loads(marker.read_text())
    if manifest["lock_sha256"] != sha(lock_path):
        raise ValueError("toolchain lock mismatch")
    for relative, expected in manifest["files"].items():
        if sha(target / relative) != expected:
            raise ValueError("toolchain file mismatch: " + relative)
else:
    if target.exists():
        raise ValueError("partial toolchain directory; preserve it before a fresh bootstrap")
    base.mkdir(exist_ok=True)
    archives = base / "archives"
    archives.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        staged = Path(tmp) / "merged"
        staged.mkdir()
        for component in lock["components"]:
            archive = archives / (component["name"] + ".tar.xz")
            if not archive.exists():
                partial = archive.with_suffix(".partial")
                urllib.request.urlretrieve(component["url"], partial)
                partial.replace(archive)
            if sha(archive) != component["sha256"]:
                raise ValueError("toolchain archive hash mismatch")
            unpacked = Path(tmp) / component["name"]
            unpacked.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(unpacked, filter="data")
            entries = list(unpacked.iterdir())
            if len(entries) != 1 or not entries[0].is_dir():
                raise ValueError("unexpected CUDA archive layout")
            shutil.copytree(entries[0], staged, dirs_exist_ok=True, symlinks=True)
        if not (staged / "lib64").exists():
            (staged / "lib64").symlink_to("lib", target_is_directory=True)
        staged.rename(target)
    manifest = {"schema_version": 1, "lock_sha256": sha(lock_path), "files": {}}
    for path in sorted(target.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(target))] = sha(path)
    marker.write_text(json.dumps(manifest, indent=2) + "\n")

version = subprocess.check_output([str(target / "bin/nvcc"), "--version"], text=True)
if "V" + lock["nvcc_version"] not in version:
    raise ValueError("unexpected compiler version")
# Compile a trivial H100 cubin without loading it or accessing any GPU.
with tempfile.TemporaryDirectory(dir=base) as tmp:
    source = Path(tmp) / "probe.cu"
    source.write_text(
        "#include <curand.h>\n#include <curand_kernel.h>\n__global__ void toolchain_probe() {}\n"
    )
    subprocess.run(
        [
            str(target / "bin/nvcc"),
            "--cubin",
            "-arch=sm_90a",
            "--compress-mode=size",
            str(source),
            "-o",
            str(Path(tmp) / "probe.cubin"),
        ],
        check=True,
    )
print(
    json.dumps(
        {"status": "compiler_verified_no_inference", "path": str(target), "version": version}
    )
)

# Compile the exact startup sampling extension for H100, without loading or executing it.
probe_env = {
    **os.environ,
    "CUDA_HOME": str(target),
    "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    "FLASHINFER_CUDA_ARCH_LIST": "9.0a",
    "FLASHINFER_WORKSPACE_BASE": str(base / ("compiler-probe-" + sha(lock_path))),
    "MAX_JOBS": "4",
}
subprocess.run(
    [
        sys.executable,
        "-c",
        "from flashinfer.jit.sampling import gen_sampling_module; gen_sampling_module().build()",
    ],
    env=probe_env,
    check=True,
)
print(json.dumps({"status": "flashinfer_sampling_compiled_no_inference"}))
