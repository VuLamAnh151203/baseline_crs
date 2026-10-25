# TRIP baseline trên cùng setting của `crs_www`

Implementation này thêm **TRIP-adapted** như baseline thứ năm mà không sửa TDRL, PPDPP,
DiffTOD hay runner hiện hữu. Mã mới nằm trong `src/tdrl/baselines/trip/`; runner và cấu hình
riêng là `src/tdrl/comparison_trip.py` và `configs/comparison.trip.json`.

Trạng thái hiện tại là code/smoke-test readiness, **không phải kết quả khoa học**. Asset lock
`configs/trip.models.lock.json` cố ý chặn scientific run cho đến khi `bert-base-uncased` được
bootstrap bằng commit SHA bất biến và smoke H100 hoàn tất.

## Phần giữ nguyên và phần thích nghi

- Lõi TRIP vẫn có hai BERT encoder, forward/backward Transformer decoder, hai token-level CE
  losses, batch-level bidirectional Euclidean agreement và target contrastive loss.
- Inference sinh hai tập beam; forward beam bắt buộc chứa target, backward beam bắt đầu từ target;
  backward score được trừ `lambda_scale * average_bidirectional_distance`.
- DuRecDial2.0 khôi phục action/topic path từ `goal` và marker turn trong pinned raw archive.
- TopDial không có gold intermediate paths, nên dùng weak action labels hiện có và deterministic
  longest entity match trên permitted knowledge. Artifact ghi `weak_path_v1`; không được gọi đây
  là exact reproduction của TRIP trên TopDial.
- TRIP lập kế hoạch lại từ observed state ở mỗi turn. Qwen local chung của dự án hiện thực hóa
  next action/topic; GPT-2 realizer cũ không được dùng trong bảng chính.
- Không có reinforcement learning trong TRIP.

## 1. Bootstrap BERT trước allocation

Bootstrap là bước duy nhất cần mạng. Từ thư mục `tdrl`:

```bash
uv sync --project runtime --locked --python 3.11
runtime/.venv/bin/python scripts/trip/bootstrap_assets.py \
  --revision BERT_BASE_UNCASED_FULL_COMMIT_SHA \
  --output models/trip \
  --lock configs/trip.models.lock.json
```

Snapshot sẽ nằm tại:

```text
models/trip/bert-base-uncased/BERT_BASE_UNCASED_FULL_COMMIT_SHA/
```

Review và commit asset lock trước scientific run. Mọi lệnh H100 đặt `HF_HUB_OFFLINE=1` và
`TRANSFORMERS_OFFLINE=1`; runtime không gọi hosted inference.

## 2. Chuẩn bị train/dev corpus

TopDial:

```bash
runtime/.venv/bin/python -m tdrl.baselines.trip.cli prepare \
  --input data/processed/TopDial/train.jsonl \
  --output artifacts/trip-data/TopDial/train.jsonl \
  --dataset TopDial --split train

runtime/.venv/bin/python -m tdrl.baselines.trip.cli prepare \
  --input data/processed/TopDial/dev.jsonl \
  --output artifacts/trip-data/TopDial/dev.jsonl \
  --dataset TopDial --split dev
```

DuRecDial2.0 bắt buộc truyền pinned official archive để không đoán goal path:

```bash
runtime/.venv/bin/python -m tdrl.baselines.trip.cli prepare \
  --input data/processed/DuRecDial2.0/train.jsonl \
  --output artifacts/trip-data/DuRecDial2.0/train.jsonl \
  --dataset DuRecDial2.0 --split train \
  --source-archive data/raw/DuRecDial2.0/data.zip

runtime/.venv/bin/python -m tdrl.baselines.trip.cli prepare \
  --input data/processed/DuRecDial2.0/dev.jsonl \
  --output artifacts/trip-data/DuRecDial2.0/dev.jsonl \
  --dataset DuRecDial2.0 --split dev \
  --source-archive data/raw/DuRecDial2.0/data.zip
```

Preparation chỉ nhận `train`/`dev`. Test reference turns không có đường code đi vào fitting.

## 3. Smoke test

CPU smoke chỉ kiểm tra tensor/control flow, không phải H100 evidence:

```bash
runtime/.venv/bin/python -m tdrl.baselines.trip.cli smoke \
  --output artifacts/trip-smoke-cpu
```

H100 smoke:

```bash
mkdir -p logs
export TRIP_BERT_REVISION=BERT_BASE_UNCASED_FULL_COMMIT_SHA
sbatch --parsable scripts/slurm/trip_smoke.sbatch
```

H100 smoke có hai gate tuần tự: gate đầu nạp snapshot BERT thật, dựng full
dual-encoder/dual-decoder TRIP rồi chạy BF16 forward, backward, loss và constrained decoding;
gate sau khởi động đồng thời hai Qwen server, giữ TRIP trên cùng H100, gọi live 4096-D embedding
và hoàn tất một interactive dialogue turn. Tiny CPU smoke không thay thế hai bước này.

## 4. Train checkpoint riêng cho mỗi dataset

Đặt biến `REV` bằng SHA đã khóa. Ví dụ TopDial:

```bash
REV=BERT_BASE_UNCASED_FULL_COMMIT_SHA
sbatch --parsable scripts/slurm/trip_stage.sbatch \
  runtime/.venv/bin/python -m tdrl.baselines.trip.cli train \
  --config configs/trip/topdial.json \
  --train artifacts/trip-data/TopDial/train.jsonl \
  --dev artifacts/trip-data/TopDial/dev.jsonl \
  --encoder models/trip/bert-base-uncased/$REV \
  --tokenizer models/trip/bert-base-uncased/$REV \
  --model-lock configs/trip.models.lock.json \
  --output artifacts/trip-train/TopDial
```

DuRecDial2.0 dùng `configs/trip/durecdial.json` và output riêng. Resume bằng
`--resume artifacts/trip-train/DATASET/checkpoint-last.pt`; checkpoint kiểm tra training hash và
khôi phục optimizer, scheduler, Python/NumPy/Torch/CUDA RNG cùng exact batch cursor.

## 5. Chọn matched budget trên dev

Natural config dùng beam 3, tối đa 80 plan tokens, lambda 3.0. Tạo một lưới nhỏ chỉ thay đổi
decoding compute, ví dụ:

```bash
runtime/.venv/bin/python scripts/trip/make_matched_config.py \
  --base configs/trip/topdial.json \
  --output artifacts/trip-configs/topdial-b2-l48.json \
  --beam-size 2 --max-plan-tokens 48
```

Chạy các candidate trên cùng dev scenarios/seeds và chọn candidate có H100-seconds gần budget
đích nhất. Không chọn theo test performance. Script từ chối thay model/training signature và
không ghi đè candidate cũ.

## 6. Evaluate dev

```bash
REV=BERT_BASE_UNCASED_FULL_COMMIT_SHA
sbatch --parsable scripts/slurm/trip_stage.sbatch \
  runtime/.venv/bin/python -m tdrl.baselines.trip.evaluate_cli evaluate \
  --root . --output artifacts/comparison-trip \
  --config configs/comparison.trip.json \
  --checkpoint-topdial artifacts/trip-train/TopDial/checkpoint-final.pt \
  --checkpoint-durecdial artifacts/trip-train/DuRecDial2.0/checkpoint-final.pt \
  --natural-config-topdial configs/trip/topdial.json \
  --matched-config-topdial artifacts/trip-configs/TOPDIAL_SELECTED.json \
  --natural-config-durecdial configs/trip/durecdial.json \
  --matched-config-durecdial artifacts/trip-configs/DUREC_SELECTED.json \
  --encoder models/trip/bert-base-uncased/$REV \
  --tokenizer models/trip/bert-base-uncased/$REV \
  --model-lock configs/trip.models.lock.json --split dev
```

Runner sử dụng cùng Qwen realizer, simulator, acceptance rules, seeds 17/29/43, profile/resistant
protocols và turn budget 10. TRIP weights được nạp tuần tự theo dataset; mỗi turn lưu action/topic,
beam counts, sequence forwards, agreement distance, latency và checkpoint/config hashes.

## 7. Khóa selection rồi mới mở test

Sau khi dev report hoàn chỉnh, chạy `lock-selection` với cùng arguments artifact, cộng:

```text
--development-evidence artifacts/comparison-trip/dev/trip-report.json
--base-selection artifacts/comparison-difftod/selection.lock.json
--output artifacts/comparison-trip/selection.lock.json
```

Test dùng lại lệnh `evaluate`, thay `--split test` và thêm
`--selection-lock artifacts/comparison-trip/selection.lock.json`. Runner từ chối test nếu bất kỳ
checkpoint, natural/matched config, asset lock hay comparison config khác selection.

## Kiểm tra trước H100/commit

```bash
python -m compileall -q src tests scripts/trip
ruff check src/tdrl/baselines/trip src/tdrl/comparison_trip.py tests/test_trip.py scripts/trip
pytest -q tests/test_trip.py
git status --short --branch
```

CPU smoke/unit tests không chứng minh runtime H100. Scientific run chỉ hợp lệ sau asset bootstrap,
Gate 2 H100, development budget selection và immutable selection lock.
