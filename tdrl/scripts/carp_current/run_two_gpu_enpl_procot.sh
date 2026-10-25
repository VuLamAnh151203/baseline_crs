#!/usr/bin/env bash
set -euo pipefail

project_root="${CARP_BASELINE_ROOT:-/home/u26anhvl/crs/baseline_crs/tdrl}"
carp_root="${CARP_CURRENT_ROOT:-/home/u26anhvl/crs/carp_www}"
run_root="${CARP_CURRENT_RUN_ROOT:-/mnt/data/u26anhvl/crs/carp_eval/carp-current-e5bf703-full-v2-62061}"
count="${CARP_CURRENT_COUNT:-200}"
workers="${CARP_CURRENT_WORKERS_PER_LANE:-8}"
python="${CARP_PYTHON:-/home/u26anhvl/miniconda3/envs/dpdp/bin/python}"
model_root="${CARP_MODEL_ROOT:-/mnt/data/duclm3/crs_refactor/tdrl/models}"
generation_path="${CARP_GEN_MODEL_PATH:-${model_root}/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
bic_site="${CARP_VLLM_BIC_SITE:-/mnt/data/duclm3/carp_www/artifacts/vllm_bic/site}"
cuda_home="${CARP_CUDA_HOME:-/mnt/data/duclm3/crs_refactor/tdrl/toolchains/cuda-12.9.1}"
enpl_artifact="${CARP_ENPL_ARTIFACT:-/mnt/data/u26anhvl/crs/enpl/enpl-fixed1000-60429}"
port_enpl="${CARP_ENPL_PORT:-18100}"
port_procot="${CARP_PROCOT_PORT:-18200}"

mkdir -p -- "$run_root/allocation" "$run_root/control" "$run_root/logs" "$run_root/quality"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUDA_HOME="$cuda_home"
export PATH="$cuda_home/bin:$(dirname "$python"):$PATH"
export FLASHINFER_WORKSPACE_BASE="$run_root/kernel-cache"
export CARP_GENERATION_MODEL="Qwen/Qwen3.5-9B"
export PYTHONPATH="$project_root/src:$carp_root/src${PYTHONPATH:+:$PYTHONPATH}"

timestamp() { date --iso-8601=seconds; }
write_state() {
  local status="$1"
  local detail="$2"
  local tmp="$run_root/control/state.tsv.tmp-${SLURM_JOB_ID}"
  {
    printf 'status\t%s\n' "$status"
    printf 'detail\t%s\n' "$detail"
    printf 'job_id\t%s\n' "$SLURM_JOB_ID"
    printf 'run_root\t%s\n' "$run_root"
    printf 'updated_at\t%s\n' "$(timestamp)"
  } > "$tmp"
  mv -f -- "$tmp" "$run_root/control/state.tsv"
}

git -C "$project_root/.." rev-parse HEAD > "$run_root/allocation/baseline-commit.txt"
git -C "$carp_root" rev-parse HEAD > "$run_root/allocation/carp-commit.txt"
git -C "$project_root/.." status --porcelain --untracked-files=all > "$run_root/allocation/baseline-status.txt"
git -C "$carp_root" status --porcelain --untracked-files=all > "$run_root/allocation/carp-status.txt"
if [[ -s "$run_root/allocation/baseline-status.txt" || -s "$run_root/allocation/carp-status.txt" ]]; then
  write_state FAILED "dirty checkout"
  printf 'Both checkouts must be clean and committed.\n' >&2
  exit 3
fi
scontrol show job "$SLURM_JOB_ID" > "$run_root/allocation/slurm-job.txt"
nvidia-smi -L > "$run_root/allocation/gpus.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
  --format=csv,noheader > "$run_root/allocation/gpus-before.txt"

asset_manifest="$run_root/asset-manifest.json"
write_state VERIFYING "hashing evaluator, datasets, model and EnPL bank"
"$python" "$project_root/scripts/carp_current/run.py" verify-assets \
  --root "$project_root" --carp-root "$carp_root" --enpl-artifact "$enpl_artifact" \
  --model-root "$model_root" --output "$asset_manifest"

enpl_pid=""
procot_pid=""
cleanup_servers() {
  for pid in "$enpl_pid" "$procot_pid"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
  for pid in "$enpl_pid" "$procot_pid"; do
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
  enpl_pid=""
  procot_pid=""
}
trap cleanup_servers EXIT

wait_ready() {
  local port="$1"
  local pid="$2"
  local waited=0
  until curl --silent --fail "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'Server on port %s exited before readiness.\n' "$port" >&2
      return 1
    fi
    if (( waited >= 900 )); then
      printf 'Timed out waiting for server on port %s.\n' "$port" >&2
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

start_server() {
  local gpu="$1"
  local port="$2"
  local mode="$3"
  local log="$4"
  local extra_env=()
  if [[ "$mode" == "bic" ]]; then
    extra_env=(VLLM_BATCH_INVARIANT=1 PYTHONPATH="$bic_site${PYTHONPATH:+:$PYTHONPATH}")
  fi
  env CUDA_VISIBLE_DEVICES="$gpu" "${extra_env[@]}" \
    "$python" -m vllm.entrypoints.openai.api_server \
      --model "$generation_path" --served-model-name "$CARP_GENERATION_MODEL" \
      --host 127.0.0.1 --port "$port" --dtype bfloat16 --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.85 --max-model-len 16384 --max-num-seqs 256 \
      --max-num-batched-tokens 32768 --no-enable-log-requests \
      --no-enable-prefix-caching --language-model-only > "$log" 2>&1 &
  printf '%s' "$!"
}

start_pair() {
  local mode="$1"
  write_state STARTING_SERVERS "$mode generation servers on GPU0/GPU1"
  enpl_pid="$(start_server 0 "$port_enpl" "$mode" "$run_root/logs/vllm-enpl-${mode}.log")"
  procot_pid="$(start_server 1 "$port_procot" "$mode" "$run_root/logs/vllm-procot-${mode}.log")"
  wait_ready "$port_enpl" "$enpl_pid"
  wait_ready "$port_procot" "$procot_pid"
}

evaluate_one() {
  local method="$1"
  local dataset="$2"
  local gpu="$3"
  local port="$4"
  local output="$run_root/$method/$dataset"
  mkdir -p -- "$output"
  env CUDA_VISIBLE_DEVICES="$gpu" "$python" "$project_root/scripts/carp_current/run.py" evaluate \
    --root "$project_root" --carp-root "$carp_root" --enpl-artifact "$enpl_artifact" \
    --asset-manifest "$asset_manifest" --method "$method" --dataset "$dataset" \
    --output "$output" --count "$count" --workers "$workers" \
    --generation-url "http://127.0.0.1:${port}/v1" \
    --embedding-url "http://127.0.0.1:9/v1" \
    > "$run_root/logs/${method}-${dataset}.log" 2>&1
}

run_phase() {
  local dataset="$1"
  write_state RUNNING_DIALOGUES "$dataset: EnPL on GPU0; ProCoT on GPU1"
  evaluate_one enpl_domain3 "$dataset" 0 "$port_enpl" &
  local eval_enpl_pid=$!
  evaluate_one procot "$dataset" 1 "$port_procot" &
  local eval_procot_pid=$!
  local status_enpl=0
  local status_procot=0
  wait "$eval_enpl_pid" || status_enpl=$?
  wait "$eval_procot_pid" || status_procot=$?
  if [[ "$status_enpl" -ne 0 || "$status_procot" -ne 0 ]]; then
    write_state FAILED "$dataset: EnPL=$status_enpl ProCoT=$status_procot"
    return 1
  fi
}

quality_one() {
  local dataset="$1"
  local port="$2"
  if [[ "$dataset" == "TopDial" ]]; then
    local file="topdial_test200_seed2911.jsonl"
    local kind="topdial"
  else
    local file="durecdial_test200_seed2911.jsonl"
    local kind="durecdial"
  fi
  "$python" -m carp.cli quality \
    --run "enpl_domain3=$run_root/enpl_domain3/$dataset/results.jsonl" \
    --run "procot=$run_root/procot/$dataset/results.jsonl" \
    --dataset-path "$carp_root/data/eval/$file" --dataset-kind "$kind" \
    --out-dir "$run_root/quality/$dataset" --model "$CARP_GENERATION_MODEL" \
    --llm-base-url "http://127.0.0.1:${port}/v1" --workers 16 \
    > "$run_root/logs/quality-${dataset}.log" 2>&1
}

start_pair bic
# Global barrier: neither TopDial shard starts until both DuRecDial shards pass.
run_phase DuRecDial2.0
run_phase TopDial
cleanup_servers

start_pair stock
write_state RUNNING_QUALITY "TopDial Ovr on GPU0; DuRecDial Ovr on GPU1"
quality_one TopDial "$port_enpl" &
quality_top_pid=$!
quality_one DuRecDial2.0 "$port_procot" &
quality_durec_pid=$!
quality_top_status=0
quality_durec_status=0
wait "$quality_top_pid" || quality_top_status=$?
wait "$quality_durec_pid" || quality_durec_status=$?
if [[ "$quality_top_status" -ne 0 || "$quality_durec_status" -ne 0 ]]; then
  write_state FAILED "quality: TopDial=$quality_top_status DuRecDial=$quality_durec_status"
  exit 1
fi

write_state FINALIZING "validating counts, duplicates, errors and quality coverage"
"$python" "$project_root/scripts/carp_current/run.py" finalize \
  --asset-manifest "$asset_manifest" --run-root "$run_root" --count "$count"
write_state COMPLETE "EnPL-domain3 and ProCoT CARP-current fixed-400 complete"
printf 'COMPLETE %s\n' "$run_root"
