#!/usr/bin/env bash
# Keep source code in the checkout while placing large corpora and model weights on /mnt/data.
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
default_storage_root="/mnt/data/duclm3/crs_refactor/tdrl"
data_root="${1:-$default_storage_root/data}"
model_root="${2:-$default_storage_root/models}"

for target in "$data_root" "$model_root"; do
  if [[ ! "$target" =~ ^/[A-Za-z0-9_./-]+$ || "$target" == *"/../"* || "$target" == */.. ]]; then
    printf 'Storage path must be a normalized absolute path: %s\n' "$target" >&2
    exit 2
  fi
done

mkdir -p "$data_root/raw" "$data_root/processed" "$model_root"

link_storage_directory() {
  local link_path="$1"
  local target_path="$2"
  mkdir -p "$(dirname "$link_path")"

  if [[ -L "$link_path" ]]; then
    if [[ "$(readlink -f "$link_path")" == "$(readlink -f "$target_path")" ]]; then
      return 0
    fi
    printf 'Refusing to change existing symlink: %s -> %s\n' "$link_path" "$(readlink "$link_path")" >&2
    return 2
  fi

  if [[ -e "$link_path" ]]; then
    # Empty ignored directories from a fresh checkout can be replaced safely; never move or
    # delete a non-empty directory because it may contain the only copy of data or weights.
    if [[ -d "$link_path" ]] && rmdir "$link_path" 2>/dev/null; then
      :
    else
      printf 'Refusing to replace existing path; migrate/verify it first: %s\n' "$link_path" >&2
      return 2
    fi
  fi

  ln -s "$target_path" "$link_path"
}

link_storage_directory "$project_root/data/raw" "$data_root/raw"
link_storage_directory "$project_root/data/processed" "$data_root/processed"
link_storage_directory "$project_root/models" "$model_root"

printf 'raw data:       %s\n' "$data_root/raw"
printf 'processed data: %s\n' "$data_root/processed"
printf 'models:         %s\n' "$model_root"
