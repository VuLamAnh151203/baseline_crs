# DPDP baseline trên cùng setting của `crs_www`

Implementation này thêm **DPDP-adapted** trong namespace riêng
`src/tdrl/baselines/dpdp/`. Nó không sửa code PPDPP, DiffTOD, TRIP hay TDRL và không sửa
repository nguồn `../DPDP`.

Trạng thái sau commit là **code/CPU-smoke readiness**, chưa phải kết quả khoa học. Scientific
run chỉ hợp lệ sau khi pin RoBERTa-large, chạy đủ H100 smoke, hoàn tất offline RL + MCTS
self-play, chọn gate/search budget trên development và tạo immutable selection lock.

## Fidelity và adaptation

- Nguồn tham chiếu: `https://github.com/cs-holder/DPDP`, revision
  `4d1e22e35db452d51f0f39388456d310ec1c971b`.
- Shared `FacebookAI/roberta-large` encoder, actor head và action-value Q head.
- Offline objective giữ executable behavior của released code:
  `-log pi(a|s) * stop_gradient(return-Q(s,a)) + critic_weight*MSE(Q(s,a),return)`.
- Self-play luôn thực thi action do open-loop PUCT MCTS chọn và chỉ update trên trajectory đã
  thực thi; các nhánh nội bộ không được đưa trực tiếp vào batch train.
- Inference dùng confidence gap top-1/top-2 để chọn System 1 hoặc System 2. System 2 chọn root
  action theo visit count và dùng cached root realization tốt nhất.
- Qwen rollout user chỉ tồn tại trong tree. Actual evaluation user vẫn do shared
  `run_dialogue()` sinh. Shared versioned local-Qwen semantic judge là nguồn success duy nhất.
- TopDial/DuRecDial dùng complete-trajectory adaptation; action labels là weak labels đã khai báo
  của canonical corpus. Đây không phải exact reproduction trên ESConv/CIMA/CraigslistBargain.
- `configs/dpdp.fidelity.json` ghi rõ discrepancy paper/code và adaptation contract.

## 1. Tạo Miniconda environment trên login node

Không tạo environment hoặc cài package trong Slurm job. Dùng Python 3.11 và sync exact runtime
lock vào environment `dpdp`:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n dpdp -c conda-forge --override-channels python=3.11 pip
conda activate dpdp
python -m pip install uv==0.8.13
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync \
  --project runtime --locked --active --inexact
```

Model/data/toolchain dùng cùng shared paths với TDRL. Mọi DPDP run artifact, checkpoint và
evaluation output dùng physical root `/mnt/data/u26anhvl/crs/dpdp`; không load scientific
checkpoint qua symlink `artifacts`. Mỗi full allocation dùng một thư mục `deadline250-JOB_ID`
riêng.

## 2. Kiểm tra CPU

Không chạy test trên login node. Submit CPU gate nhẹ:

```bash
mkdir -p /mnt/data/u26anhvl/crs/dpdp/slurm
sbatch --parsable scripts/slurm/dpdp_cpu_gate.sbatch
```

CPU smoke chỉ kiểm tra tensor/control flow với tiny components, không phải H100 evidence.

## 3. Bootstrap immutable RoBERTa-large trước allocation

Đây là bước cần mạng; inference/training sau đó đặt offline mode:

```bash
python scripts/dpdp/bootstrap_assets.py \
  --revision ROBERTA_LARGE_FULL_40_CHAR_COMMIT_SHA \
  --output models \
  --lock configs/dpdp.models.lock.json
```

Snapshot nằm tại:

```text
models/roberta-large/REVISION/
```

Review và commit asset lock trước scientific run. Application code không dùng hosted inference
hay credential; Qwen chỉ được gọi qua loopback-only runtime của dự án.

## 4. Submit DPDP-250 E2E trong một allocation 2 H100 / 6 ngày

`dpdp_stage.sbatch` xin một Job ID với hai H100, 16 CPU, 160G RAM và walltime 6 ngày. TopDial và
DuRecDial2.0 chạy song song trên hai Slurm steps độc lập qua reward annotation, offline training và
MCTS self-play. Mỗi dataset dùng `5 x 50 = 250` self-play cases; simulations/depth/critic samples
vẫn là `10/5/10`. Các hyperparameter phase-specific giữ launch setting gốc: offline dropout
`0.1`, 10 epochs, batch 8, learning rate `6e-6`, weight decay `0.001`, 400 warmup steps và seed
42; self-play dropout `0.25`, learning rate `1e-6` và seed 1. Sau khi cả hai checkpoint hoàn tất,
cùng allocation chạy development, ghi global
immutable selection lock, rồi mở prefix đầu tiên của resource fixed-1000: 200 TopDial + 200
DuRecDial2.0. Official test chỉ chạy natural budget, cả `profile` và `resistant`, tổng cộng 800
DPDP dialogues. Hai dataset chạy song song và được merge thành report prefix-400 riêng.

Worktree phải sạch và đã commit trước khi job bắt đầu. Script tự tạo physical run root dưới
`/mnt/data`:

```bash
JOB_ID="$(bash scripts/submit_dpdp_deadline.sh | sed -n 's/^Submitted DPDP-250 E2E job: //p')"
export DPDP_STORAGE_ROOT=/mnt/data/u26anhvl/crs/dpdp
export DPDP_RUN_ROOT="$DPDP_STORAGE_ROOT/deadline250-$JOB_ID"
```

Hoặc chạy wrapper trực tiếp và copy Job ID mà nó in ra:

```bash
bash scripts/submit_dpdp_deadline.sh
```

`DPDP_FIXED_TEST_COUNT` mặc định là `400`. Sau khi prefix-400 hoàn tất, có thể dùng lại đúng
`DPDP_RUN_ROOT` và selection lock đã freeze để mở rộng lên 1000 cases mà không đổi request IDs
hoặc chạy lại dialogue đã hoàn tất:

```bash
export DPDP_RUN_ROOT=/mnt/data/u26anhvl/crs/dpdp/deadline250-FIRST_JOB_ID
export DPDP_FIXED_TEST_COUNT=1000
sbatch --parsable scripts/slurm/dpdp_stage.sbatch
```

Prefix-1000 bổ sung 300 TopDial + 300 DuRecDial2.0; report và E2E summary của prefix-400 không
bị ghi đè.

Scientific preflight yêu cầu worktree sạch và đã commit. Job fail trước heavy compute nếu provenance
này không hợp lệ. Resume phải dùng explicit wrapper và đúng physical run root ban đầu; wrapper từ chối
symlink, path ngoài namespace DPDP, hoặc run root không có evidence cũ:

```bash
bash scripts/resume_dpdp_deadline.sh \
  /mnt/data/u26anhvl/crs/dpdp/deadline250-ORIGINAL_JOB_ID
```

Không chạy lại `submit_dpdp_deadline.sh` theo fresh mode để recovery vì Job ID mới sẽ tạo run root mới.
Khi resume, completed stage được kiểm tra hash/checkpoint rồi skip; reward requests đọc lại journal;
offline training đọc model/optimizer/scheduler/RNG và exact batch cursor từ `checkpoint-last.pt`; self-play
đọc exact epoch/case cursor và checkpoint mỗi 5 cases, cuối epoch hoặc khi nhận stop signal. Dialogue
development/test snapshot sau từng turn và completed dialogue được checksum-seal rồi reuse. Nếu process
dừng đúng lúc một local-model HTTP request đang in-flight, request ledger cố ý fail closed thay vì bí mật
gọi lại; giữ nguyên evidence và audit/authorize đúng request đó trước khi resume.

Log Slurm tổng nằm ở:

```text
logs/dpdp-e2e250-2gpu-JOB_ID.out
logs/dpdp-e2e250-2gpu-JOB_ID.err
```

Progress machine-readable và log từng stage nằm ở:

```text
/mnt/data/u26anhvl/crs/dpdp/deadline250-JOB_ID/status.json
/mnt/data/u26anhvl/crs/dpdp/deadline250-JOB_ID/status-TopDial.json
/mnt/data/u26anhvl/crs/dpdp/deadline250-JOB_ID/status-DuRecDial2.0.json
/mnt/data/u26anhvl/crs/dpdp/deadline250-JOB_ID/events.jsonl
/mnt/data/u26anhvl/crs/dpdp/deadline250-JOB_ID/logs/*.log
/mnt/data/u26anhvl/crs/dpdp/deadline250-JOB_ID/allocation/segment-JOB_ID.json
/mnt/data/u26anhvl/crs/dpdp/deadline250-JOB_ID/allocation/failure-JOB_ID.json
/mnt/data/u26anhvl/crs/dpdp/deadline250-JOB_ID/allocation/gpu-telemetry-JOB_ID.csv
```

`status.json` ghi stage E2E hiện tại; hai file status dataset ghi `stage_number/stage_total` và
progress. Annotation ghi
`critic_requests completed/total`; offline training ghi optimizer `step/total`, epoch và batch;
self-play ghi case/total, epoch, case index và scenario ID. Controller cập nhật heartbeat mỗi 30 giây.
Evaluation ghi từng completed/reused dialogue vào `evaluation-progress.jsonl` và cập nhật atomic
`evaluation-status.json` (completed/expected/percent) trong từng dataset worker. Mỗi lần submit/resume là
một segment riêng nên log và traceback cũ không bị ghi đè.

```bash
squeue -j "$JOB_ID" -o '%.18i %.28j %.9T %.10M %.30R'
tail -f "logs/dpdp-e2e250-2gpu-$JOB_ID.out"
watch -n 10 "python -m json.tool $DPDP_RUN_ROOT/status.json"
tail -f "$DPDP_RUN_ROOT/events.jsonl"
```

Development vẫn kiểm tra cả natural và matched proxy để giữ selection contract. `matched` được cố
định ở minimum grid `2 simulations/depth 1` và không phải official compute-matched claim vì chưa có
measured TDRL development target. Fixed-prefix official test không chạy proxy này; nó chỉ dùng
natural budget đầy đủ `10 simulations/depth 5`, giống shared fixed-1000 setting.

Các lệnh thủ công dưới đây là stage-level recovery/debug reference; không cần chạy lần lượt cho
normal full training job.

TopDial:

```bash
python -m tdrl.baselines.dpdp.cli prepare \
  --input data/processed/TopDial/train.jsonl \
  --output "$DPDP_RUN_ROOT/data/TopDial/train.jsonl" \
  --dataset TopDial --split train

python -m tdrl.baselines.dpdp.cli prepare \
  --input data/processed/TopDial/dev.jsonl \
  --output "$DPDP_RUN_ROOT/data/TopDial/dev.jsonl" \
  --dataset TopDial --split dev
```

DuRecDial2.0 thay dataset/config/path tương ứng. `prepare` chỉ nhận `train` hoặc `dev`, lưu
manifest/hash và từ chối ghi đè. Mỗi state chỉ chứa prefix đã quan sát; test bodies không được mở.

Các lệnh H100 bên dưới là stage-level reference mà controller thực thi bên trong allocation; không
chạy chúng trực tiếp trên login node và không chạy song song với controller.

## 5. Annotate reward bằng local Qwen trên H100

Chạy riêng cho train và dev của từng dataset trong **cùng persistent allocation**. Ví dụ TopDial
train với allocation đã đăng ký:

```bash
python -m tdrl.baselines.dpdp.cli annotate-rewards \
  --root . --runtime-config configs/runtime.json \
  --config configs/dpdp/topdial-deadline250.json \
  --input "$DPDP_RUN_ROOT/data/TopDial/train.jsonl" \
  --output "$DPDP_RUN_ROOT/data/TopDial/train.annotated.jsonl"
```

Critic được gọi 10 lần/state theo config. Chỉ bốn exact labels được chấp nhận; malformed output
retry hữu hạn rồi fail closed. Journal cho phép resume mà không gọi lại request đã hoàn tất.
Discounted return được reset ở ranh giới mỗi dialogue. Stage này chỉ khởi động generation server;
embedding model không chiếm VRAM khi không được dùng. Các critic độc lập chạy theo fixed ordered
micro-batch theo `request_concurrency`; runner DPDP-250 dùng 10 sequence đồng thời. Request ID,
seed và thứ tự artifact không đổi.

## 6. Offline actor–critic pretraining

```bash
REV=ROBERTA_LARGE_FULL_40_CHAR_COMMIT_SHA
MODEL="models/roberta-large/$REV"
python -m tdrl.baselines.dpdp.cli train \
  --config configs/dpdp/topdial-deadline250.json \
  --train "$DPDP_RUN_ROOT/data/TopDial/train.annotated.jsonl" \
  --dev "$DPDP_RUN_ROOT/data/TopDial/dev.annotated.jsonl" \
  --encoder "$MODEL" --tokenizer "$MODEL" \
  --model-lock configs/dpdp.models.lock.json \
  --output "$DPDP_RUN_ROOT/offline/TopDial"
```

Resume bằng `--resume .../checkpoint-last.pt`. Checkpoint lưu exact epoch/batch cursor, model,
optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, config/data/taxonomy/model-lock hashes và source
revision. Config/provenance mismatch fail closed. `checkpoint-last.pt` chỉ được cập nhật theo
`offline_checkpoint_every`, cuối epoch hoặc khi có stop request; không ghi nhiều-GB checkpoint sau
mỗi optimizer step.

## 7. MCTS-guided self-play

```bash
python -m tdrl.baselines.dpdp.cli selfplay \
  --root . --runtime-config configs/runtime.json \
  --config configs/dpdp/topdial-deadline250.json \
  --scenarios data/processed/TopDial/train.jsonl \
  --offline-checkpoint "$DPDP_RUN_ROOT/offline/TopDial/checkpoint-final.pt" \
  --encoder "$MODEL" --tokenizer "$MODEL" \
  --model-lock configs/dpdp.models.lock.json \
  --output "$DPDP_RUN_ROOT/selfplay/TopDial"
```

Self-play chỉ đọc training scenarios. Mỗi actual self-play action đến từ MCTS; cached selected
assistant/user realization tạo executed trajectory, rồi discounted returns và actor–critic update.
Resume dùng `checkpoint-last.pt` và tiếp tục đúng case kế tiếp. MCTS simulations và cases vẫn tuần
tự để không đổi thuật toán; chỉ các critic samples độc lập được micro-batch ở concurrency 10. Stage
chỉ khởi động generation server và checkpoint mỗi 10 cases, cuối epoch hoặc stop request.

## 8. H100 smoke trước full run

```bash
mkdir -p logs
export DPDP_ROBERTA_REVISION=ROBERTA_LARGE_FULL_40_CHAR_COMMIT_SHA
sbatch --parsable scripts/slurm/dpdp_smoke.sbatch
```

Smoke gồm: full RoBERTa BF16 forward/backward; live 4096-D embedding; hai Qwen server cùng resident;
một System-2 decision với 2 simulations; internal assistant/user/critic; actual shared simulator và
acceptance evaluator. Nếu bất kỳ gate nào fail, dừng trước full training.

## 9. Gate threshold và matched budget

`gate_threshold=0.2` trong base config là candidate ban đầu, không phải selected scientific value.
Thu confidence gaps trên fixed development scenarios, khai báo threshold candidates trước, rồi tạo
config đã chọn. Matched grid chỉ thay search compute, ví dụ:

```bash
python scripts/dpdp/make_matched_config.py \
  --base "$DPDP_RUN_ROOT/configs/topdial-natural-selected.json" \
  --output "$DPDP_RUN_ROOT/configs/topdial-s4-d2.json" \
  --budget matched --simulations 4 --depth 2
```

Grid được phép: simulations `[2,4,6]`, depth `[1,2,3]`. Chọn candidate có measured development
H100-seconds/dialogue gần target nhất, không chọn theo test score. Script kiểm tra training signature
không đổi. Natural config cũng phải được materialize bằng `--budget natural --simulations 10
--depth 5` với threshold đã chọn trước khi lock.

## 10. Development evaluation và selection lock

```bash
python -m tdrl.baselines.dpdp.evaluate_cli evaluate \
  --root . --output "$DPDP_RUN_ROOT/evaluation" \
  --config configs/comparison.dpdp.json \
  --checkpoint-topdial "$DPDP_RUN_ROOT/selfplay/TopDial/checkpoint-final.pt" \
  --checkpoint-durecdial "$DPDP_RUN_ROOT/selfplay/DuRecDial2.0/checkpoint-final.pt" \
  --natural-config-topdial "$DPDP_RUN_ROOT/configs/topdial-natural-selected.json" \
  --matched-config-topdial "$DPDP_RUN_ROOT/configs/topdial-matched-selected.json" \
  --natural-config-durecdial "$DPDP_RUN_ROOT/configs/durecdial-natural-selected.json" \
  --matched-config-durecdial "$DPDP_RUN_ROOT/configs/durecdial-matched-selected.json" \
  --encoder "$MODEL" --tokenizer "$MODEL" \
  --model-lock configs/dpdp.models.lock.json --split dev
```

Sau khi report dev hoàn chỉnh, dùng cùng artifact arguments với subcommand `lock-selection`, cộng:

```text
--development-evidence "$DPDP_RUN_ROOT/evaluation/dev/dpdp-report.json"
--base-selection artifacts/comparison-trip/selection.lock.json
--output "$DPDP_RUN_ROOT/evaluation/selection.lock.json"
```

Lock chứa checkpoint/config/asset/prompt/source/development hashes. File được tạo immutable và không
ghi đè.

## 11. Fixed-1000 official-test prefix

Chạy lại `evaluate --split test` và thêm:

```text
--selection-lock "$DPDP_RUN_ROOT/evaluation/selection.lock.json"
--fixed-test-setting configs/baseline-fixed1000-test-v1.json
--fixed-test-count 400
```

Không submit một allocation mới cho từng stage. Reward annotation, offline training, self-play,
development evaluation, selection và test phải chạy tuần tự trong cùng owned H100 allocation; mỗi
stage có checkpoint/manifest để dừng an toàn nếu allocation không đủ thời gian.

Runner chỉ mở official test sau khi verify lock. Prefix-400 replay đúng 200 TopDial và 200
DuRecDial2.0 từ resource fixed-1000, dùng experimental seed `17`, protocols `profile/resistant`,
natural budget, turn budget 10, shared Qwen realization/simulator, refusal guard, semantic outcome
judge và metrics. Tổng cộng là 800 DPDP dialogues. Prefix-1000 sau đó mở rộng lên 500 cases mỗi
dataset mà không đổi selection hoặc chạy lại completed snapshots. Mỗi turn ghi policy probabilities,
Q values, confidence gap, selected system,
MCTS visits/Q/priors/depth/simulations, request/token/latency và artifact hashes.

## Giới hạn cần báo cáo

- Đây là DPDP adaptation sang conversational recommendation, không phải exact reproduction.
- Weak action labels có thể tạo label noise; provenance của label được giữ trong transition.
- CPU tests/smoke không chứng minh H100 readiness hay scientific validity.
- Không báo cáo test trước development selection lock; không đổi threshold/budget/checkpoint sau lock.
