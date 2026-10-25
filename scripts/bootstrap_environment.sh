#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
project_root="$repo_root/tdrl"
python_bin="${1:-python3.11}"

command -v uv >/dev/null 2>&1 || {
  printf 'uv is required. Install the pinned bootstrap tool first, e.g. pip install uv==0.8.13\n' >&2
  exit 2
}
uv sync --project "$project_root/runtime" --locked --python "$python_bin"
printf 'Runtime ready: %s\n' "$project_root/runtime/.venv/bin/python"
printf 'Next: link data/models with tdrl/scripts/link_h100_storage.sh.\n'
