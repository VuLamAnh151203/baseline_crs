#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s PROJECT_ROOT STORAGE_ROOT\n' "$0" >&2
  exit 2
fi

project_root="$(realpath -- "$1")"
storage_root="$(realpath -m -- "$2")"
repo_root="$(git -C "$project_root" rev-parse --show-toplevel)"
relative_project="$(realpath --relative-to="$repo_root" "$project_root")"
commit="$(git -C "$repo_root" rev-parse HEAD)"
checkout_root="${storage_root}/checkouts/${commit}"
checkout_project="${checkout_root}/${relative_project}"

if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]]; then
  printf 'Refusing DPDP checkout preparation from a dirty source worktree.\n' >&2
  git -C "$repo_root" status --short >&2
  exit 3
fi

mkdir -p -- "${storage_root}/checkouts"
if [[ ! -e "$checkout_root" ]]; then
  git -C "$repo_root" worktree add --detach "$checkout_root" "$commit" >&2
fi
if [[ ! -d "$checkout_project" ]]; then
  printf 'DPDP checkout is missing the project directory: %s\n' "$checkout_project" >&2
  exit 4
fi
if [[ "$(git -C "$checkout_root" rev-parse HEAD)" != "$commit" ]]; then
  printf 'DPDP checkout commit mismatch: %s\n' "$checkout_root" >&2
  exit 5
fi

link_exact() {
  local source_path target_path resolved_source resolved_target
  source_path="$1"
  target_path="$2"
  if [[ ! -e "$source_path" ]]; then
    printf 'Missing DPDP checkout asset: %s\n' "$source_path" >&2
    exit 6
  fi
  resolved_source="$(realpath -- "$source_path")"
  if [[ -L "$target_path" ]]; then
    resolved_target="$(realpath -- "$target_path")"
    if [[ "$resolved_target" != "$resolved_source" ]]; then
      printf 'DPDP checkout link points elsewhere: %s\n' "$target_path" >&2
      exit 7
    fi
  elif [[ -e "$target_path" ]]; then
    printf 'Refusing to replace DPDP checkout path: %s\n' "$target_path" >&2
    exit 8
  else
    mkdir -p -- "$(dirname -- "$target_path")"
    ln -s -- "$resolved_source" "$target_path"
  fi
}

link_exact "$project_root/data/raw" "$checkout_project/data/raw"
link_exact "$project_root/data/processed" "$checkout_project/data/processed"
link_exact "$project_root/models" "$checkout_project/models"
link_exact "$project_root/toolchains" "$checkout_project/toolchains"

dirty="$(git -C "$checkout_root" status --porcelain --untracked-files=all)"
if [[ -n "$dirty" ]]; then
  printf 'Prepared DPDP checkout is not clean:\n%s\n' "$dirty" >&2
  exit 9
fi

printf '%s\n' "$checkout_project"
