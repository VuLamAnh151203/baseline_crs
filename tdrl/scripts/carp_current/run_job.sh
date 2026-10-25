#!/usr/bin/env bash
set -euo pipefail

project_root="${CARP_BASELINE_ROOT:?set CARP_BASELINE_ROOT to baseline_crs/tdrl}"
carp_root="${CARP_CURRENT_ROOT:-/home/u26anhvl/crs/carp_www}"
run_root="${CARP_CURRENT_RUN_ROOT:?set CARP_CURRENT_RUN_ROOT}"
count="${CARP_CURRENT_COUNT:?set CARP_CURRENT_COUNT}"
workers="${CARP_CURRENT_WORKERS:-16}"
generation_only="${CARP_CURRENT_GENERATION_ONLY:-0}"
generation_device="${CARP_CURRENT_GENERATION_DEVICE:-0}"
generation_port="${CARP_CURRENT_GENERATION_PORT:-8000}"
embedding_port="${CARP_CURRENT_EMBEDDING_PORT:-8001}"
python="${CARP_PYTHON:-/mnt/data/duclm3/crs_refactor/tdrl/runtime/.venv/bin/python}"
model_root="${CARP_MODEL_ROOT:-/mnt/data/duclm3/crs_refactor/tdrl/models}"
generation_path="${CARP_GEN_MODEL_PATH:-${model_root}/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
embedding_path="${CARP_EMB_MODEL_PATH:-${model_root}/Qwen3-Embedding-8B/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af}"
bic_site="${CARP_VLLM_BIC_SITE:-/mnt/data/duclm3/carp_www/artifacts/vllm_bic/site}"
cuda_home="${CARP_CUDA_HOME:-/mnt/data/duclm3/crs_refactor/tdrl/toolchains/cuda-12.9.1}"
enpl_artifact="${CARP_ENPL_ARTIFACT:-/mnt/data/u26anhvl/crs/enpl/enpl-fixed1000-60429}"

mkdir -p -- "$run_root/allocation" "$run_root/logs" "$run_root/quality"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUDA_HOME="$cuda_home"
export PATH="$cuda_home/bin:$(dirname "$python"):$PATH"
export FLASHINFER_WORKSPACE_BASE="$run_root/kernel-cache"
export CARP_GENERATION_MODEL="Qwen/Qwen3.5-9B"
export CARP_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
export CARP_LLM_BASE_URL="http://127.0.0.1:${generation_port}/v1"
export CARP_EMBEDDING_BASE_URL="http://127.0.0.1:${embedding_port}/v1"
export PYTHONPATH="$project_root/src:$carp_root/src${PYTHONPATH:+:$PYTHONPATH}"

git -C "$project_root/.." rev-parse HEAD > "$run_root/allocation/baseline-commit.txt"
git -C "$carp_root" rev-parse HEAD > "$run_root/allocation/carp-commit.txt"
git -C "$project_root/.." status --porcelain --untracked-files=all > "$run_root/allocation/baseline-status.txt"
git -C "$carp_root" status --porcelain --untracked-files=all > "$run_root/allocation/carp-status.txt"
if [[ -s "$run_root/allocation/baseline-status.txt" || -s "$run_root/allocation/carp-status.txt" ]]; then
  printf 'Both baseline_crs and carp_www must be clean committed checkouts.\n' >&2
  exit 3
fi
scontrol show job "$SLURM_JOB_ID" > "$run_root/allocation/slurm-job.txt"
nvidia-smi -L > "$run_root/allocation/gpus.txt"

asset_manifest="$run_root/asset-manifest.json"
"$python" "$project_root/scripts/carp_current/run.py" verify-assets \
  --root "$project_root" --carp-root "$carp_root" --enpl-artifact "$enpl_artifact" \
  --model-root "$model_root" --output "$asset_manifest"

gen_pid=""
emb_pid=""
cleanup_servers() {
  if [[ -n "$gen_pid" ]]; then kill "$gen_pid" 2>/dev/null || true; fi
  if [[ -n "$emb_pid" ]]; then kill "$emb_pid" 2>/dev/null || true; fi
  if [[ -n "$gen_pid" ]]; then wait "$gen_pid" 2>/dev/null || true; fi
  if [[ -n "$emb_pid" ]]; then wait "$emb_pid" 2>/dev/null || true; fi
  gen_pid=""
  emb_pid=""
}
trap cleanup_servers EXIT

wait_ready() {
  local port="$1"
  local pid="$2"
  local waited=0
  until curl --silent --fail "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'vLLM server on port %s exited before readiness.\n' "$port" >&2
      return 1
    fi
    if (( waited >= 900 )); then
      printf 'Timed out waiting for vLLM server on port %s.\n' "$port" >&2
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

start_bic_servers() {
  env CUDA_VISIBLE_DEVICES="$generation_device" VLLM_BATCH_INVARIANT=1 \
    PYTHONPATH="$bic_site${PYTHONPATH:+:$PYTHONPATH}" \
    "$python" -m vllm.entrypoints.openai.api_server \
      --model "$generation_path" --served-model-name "$CARP_GENERATION_MODEL" \
      --host 127.0.0.1 --port "$generation_port" --dtype bfloat16 --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.85 --max-model-len 16384 --max-num-seqs 256 \
      --max-num-batched-tokens 32768 --no-enable-log-requests \
      --no-enable-prefix-caching --language-model-only \
      > "$run_root/logs/vllm-generation-bic.log" 2>&1 &
  gen_pid=$!
  wait_ready "$generation_port" "$gen_pid"
  if [[ "$generation_only" == "0" ]]; then
    env CUDA_VISIBLE_DEVICES=1 "$python" -m vllm.entrypoints.openai.api_server \
        --model "$embedding_path" --served-model-name "$CARP_EMBEDDING_MODEL" \
        --host 127.0.0.1 --port "$embedding_port" --dtype bfloat16 --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.40 --max-model-len 8192 --max-num-seqs 64 \
        --no-enable-log-requests --no-enable-prefix-caching --runner pooling --convert embed \
        > "$run_root/logs/vllm-embedding.log" 2>&1 &
    emb_pid=$!
    wait_ready "$embedding_port" "$emb_pid"
  else
    printf 'Embedding server omitted: EnPL/ProCoT issue zero embedding requests.\n' \
      > "$run_root/logs/vllm-embedding-not-required.log"
  fi
}

start_stock_generation() {
  env CUDA_VISIBLE_DEVICES="$generation_device" "$python" -m vllm.entrypoints.openai.api_server \
      --model "$generation_path" --served-model-name "$CARP_GENERATION_MODEL" \
      --host 127.0.0.1 --port "$generation_port" --dtype bfloat16 --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.85 --max-model-len 16384 --max-num-seqs 256 \
      --max-num-batched-tokens 32768 --no-enable-log-requests \
      --no-enable-prefix-caching --language-model-only \
      > "$run_root/logs/vllm-generation-quality-stock.log" 2>&1 &
  gen_pid=$!
  wait_ready "$generation_port" "$gen_pid"
}

evaluate_dataset() {
  local method="$1"
  local dataset="$2"
  local output="$run_root/$method/$dataset"
  mkdir -p -- "$output"
  env CUDA_VISIBLE_DEVICES="$generation_device" "$python" "$project_root/scripts/carp_current/run.py" evaluate \
    --root "$project_root" --carp-root "$carp_root" --enpl-artifact "$enpl_artifact" \
    --asset-manifest "$asset_manifest" --method "$method" --dataset "$dataset" \
    --output "$output" --count "$count" --workers "$workers" \
    --generation-url "$CARP_LLM_BASE_URL" --embedding-url "$CARP_EMBEDDING_BASE_URL" \
    > "$run_root/logs/${method}-${dataset}.log" 2>&1
}

start_bic_servers
for method in enpl_domain3 procot; do
  evaluate_dataset "$method" TopDial &
  top_pid=$!
  evaluate_dataset "$method" DuRecDial2.0 &
  durec_pid=$!
  top_status=0
  durec_status=0
  wait "$top_pid" || top_status=$?
  wait "$durec_pid" || durec_status=$?
  if [[ "$top_status" -ne 0 || "$durec_status" -ne 0 ]]; then
    printf '%s failed: TopDial=%s DuRecDial2.0=%s\n' "$method" "$top_status" "$durec_status" >&2
    exit 1
  fi
done
cleanup_servers

# CARP's published Ovr. pass uses the same Qwen model on a stock vLLM server.
start_stock_generation
for dataset in TopDial DuRecDial2.0; do
  if [[ "$dataset" == "TopDial" ]]; then
    dataset_file="topdial_test200_seed2911.jsonl"
    dataset_kind="topdial"
  else
    dataset_file="durecdial_test200_seed2911.jsonl"
    dataset_kind="durecdial"
  fi
  "$python" -m carp.cli quality \
    --run "enpl_domain3=$run_root/enpl_domain3/$dataset/results.jsonl" \
    --run "procot=$run_root/procot/$dataset/results.jsonl" \
    --dataset-path "$carp_root/data/eval/$dataset_file" --dataset-kind "$dataset_kind" \
    --out-dir "$run_root/quality/$dataset" --model "$CARP_GENERATION_MODEL" \
    --llm-base-url "$CARP_LLM_BASE_URL" --workers 32 \
    > "$run_root/logs/quality-${dataset}.log" 2>&1
done

"$python" "$project_root/scripts/carp_current/run.py" finalize \
  --asset-manifest "$asset_manifest" --run-root "$run_root" --count "$count"
printf 'COMPLETE %s\n' "$run_root"
