#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/submit_baseline.sh METHOD MODE

METHOD: direct_llm_v2 | tepl_v2 | enpl_domain3 | procot | dpdp |
        scope_crs_lite | epl_native_qwen
MODE:   mini | full400

Required environment:
  BASELINE_STORAGE_ROOT  absolute durable output root

Optional environment:
  BASELINE_PYTHON        defaults to tdrl/runtime/.venv/bin/python
  SLURM_PARTITION        defaults to main
  SLURM_ACCOUNT          defaults to normal
  SLURM_GRES             defaults to gpu:nvidia_h100_80gb_hbm3:2
  BASELINE_RUN_ROOT      explicit fresh/resume run root
  BASELINE_RESUME=1      resume DPDP in BASELINE_RUN_ROOT
  ENPL_CONVPLAN_ZIP      required for enpl_domain3
  SCOPE_ASSET_ROOT       required for scope_crs_lite
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

method="$1"
mode="$2"
case "$mode" in
  mini|full400) ;;
  *) printf 'Unsupported mode: %s\n' "$mode" >&2; exit 2 ;;
esac
case "$method" in
  direct_llm_v2|tepl_v2|enpl_domain3|procot|dpdp|scope_crs_lite|epl_native_qwen) ;;
  *) printf 'Unsupported method: %s\n' "$method" >&2; usage >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
project_root="$repo_root/tdrl"
storage_root="${BASELINE_STORAGE_ROOT:?Set BASELINE_STORAGE_ROOT to an absolute durable path}"
python_bin="${BASELINE_PYTHON:-$project_root/runtime/.venv/bin/python}"

for path in "$repo_root" "$storage_root" "$python_bin"; do
  if [[ "$path" != /* || "$path" == *'/../'* || "$path" == */.. ]]; then
    printf 'Expected a normalized absolute path: %s\n' "$path" >&2
    exit 2
  fi
done
if [[ ! -x "$python_bin" ]]; then
  printf 'Runtime Python is missing or not executable: %s\n' "$python_bin" >&2
  exit 2
fi
if [[ ! -f "$project_root/pyproject.toml" ]]; then
  printf 'Invalid checkout; missing %s\n' "$project_root/pyproject.toml" >&2
  exit 2
fi

case "$method" in
  direct_llm_v2|procot) walltime="12:00:00" ;;
  enpl_domain3) walltime="18:00:00" ;;
  *) walltime="6-00:00:00" ;;
esac
if [[ "$mode" == mini ]]; then
  walltime="04:00:00"
fi

if [[ "$method" == enpl_domain3 ]]; then
  : "${ENPL_CONVPLAN_ZIP:?Set ENPL_CONVPLAN_ZIP for EnPL-domain3}"
  [[ -f "$ENPL_CONVPLAN_ZIP" ]] || { printf 'Missing ConvPlan archive: %s\n' "$ENPL_CONVPLAN_ZIP" >&2; exit 2; }
fi
if [[ "$method" == scope_crs_lite ]]; then
  : "${SCOPE_ASSET_ROOT:?Set SCOPE_ASSET_ROOT for SCOPE-CRS-lite}"
  [[ -d "$SCOPE_ASSET_ROOT" ]] || { printf 'Missing SCOPE asset root: %s\n' "$SCOPE_ASSET_ROOT" >&2; exit 2; }
fi
if [[ "${BASELINE_RESUME:-0}" == 1 && -z "${BASELINE_RUN_ROOT:-}" ]]; then
  printf 'BASELINE_RESUME=1 requires BASELINE_RUN_ROOT\n' >&2
  exit 2
fi

mkdir -p -- "$storage_root/slurm"
export BASELINE_REPO_ROOT="$repo_root"
export BASELINE_METHOD="$method"
export BASELINE_MODE="$mode"
export BASELINE_STORAGE_ROOT="$storage_root"
export BASELINE_PYTHON="$python_bin"
if [[ -n "${ENPL_CONVPLAN_ZIP:-}" ]]; then
  export ENPL_CONVPLAN_ZIP
fi
if [[ -n "${SCOPE_ASSET_ROOT:-}" ]]; then
  export SCOPE_ASSET_ROOT
fi
if [[ -n "${EPL_NATIVE_MINI_ROOT:-}" ]]; then
  export EPL_NATIVE_MINI_ROOT
fi

job_id="$(sbatch --parsable \
  --job-name="${method}-${mode}" \
  --partition="${SLURM_PARTITION:-main}" \
  --account="${SLURM_ACCOUNT:-normal}" \
  --nodes=1 --ntasks=1 \
  --gres="${SLURM_GRES:-gpu:nvidia_h100_80gb_hbm3:2}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --mem="${SLURM_MEMORY:-160G}" \
  --time="$walltime" \
  --signal=B:USR1@900 \
  --output="$storage_root/slurm/%x-%j.out" \
  --error="$storage_root/slurm/%x-%j.err" \
  --export=ALL \
  "$repo_root/scripts/slurm/baseline_e2e.sbatch")"

printf 'Submitted %s %s: %s\n' "$method" "$mode" "$job_id"
printf 'Logs: %s/slurm/%s-%s-%s.{out,err}\n' "$storage_root" "$method" "$mode" "$job_id"
