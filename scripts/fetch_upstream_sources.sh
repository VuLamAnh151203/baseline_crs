#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
destination="${1:-$repo_root/external}"
if [[ "$destination" != /* || "$destination" == *'/../'* || "$destination" == */.. ]]; then
  printf 'Destination must be a normalized absolute path: %s\n' "$destination" >&2
  exit 2
fi
mkdir -p -- "$destination"

fetch_source() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local checkout="$destination/$name"
  if [[ -e "$checkout" && ! -d "$checkout/.git" ]]; then
    printf 'Refusing to replace non-Git path: %s\n' "$checkout" >&2
    return 2
  fi
  if [[ ! -d "$checkout/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$url" "$checkout"
  fi
  git -C "$checkout" remote set-url origin "$url"
  git -C "$checkout" fetch --depth=1 origin "$revision"
  git -C "$checkout" checkout --detach "$revision"
  test "$(git -C "$checkout" rev-parse HEAD)" = "$revision"
  printf 'Pinned %-18s %s\n' "$name" "$revision"
}

fetch_source DPDP https://github.com/cs-holder/DPDP.git 4d1e22e35db452d51f0f39388456d310ec1c971b
fetch_source EPL https://github.com/huyquangdao/EPL.git 364c6fd17f1bbdb2abcac9cd2381a29bf1bd5dda
fetch_source ConvPlan https://github.com/pandazzh2020/ConvPlan.git a76d5a7d6ea1d9c7a07a9574f4901ff79ed440bd
fetch_source LLM-Proactive https://github.com/dengyang17/LLM-Proactive.git 77ff4ed7dfda11afd719db3bd40886a1a51edf4e
fetch_source convo-plan-SCOPE https://github.com/chenzhiliang94/convo-plan-SCOPE.git b35dbc64808afce25e8f8e5d6838ecf271ec4d0b
fetch_source TRIP https://github.com/iwangjian/TRIP.git ee845a5306902ae673e92083711b283b9f33bb9f
fetch_source DiffTOD https://github.com/ninglab/DiffTOD.git fd46b2887a9c0d9e0e413623e79c72ee32b9b7c1
fetch_source carp_www https://github.com/kuduck192/carp_www.git 86f56b07e4efd4e645a51e389423bea9d0d185be
