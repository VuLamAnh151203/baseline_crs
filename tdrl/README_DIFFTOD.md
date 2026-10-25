# DiffTOD baseline trên cùng setting của `crs_www`

Phần mở rộng này đưa DiffTOD vào evaluator hiện có mà không sửa planner TDRL, PPDPP hay
runner ba phương pháp cũ. Toàn bộ implementation nằm trong
`src/tdrl/baselines/difftod/`; extension tạo evidence DiffTOD cho so sánh bốn phương pháp dùng
file riêng
`src/tdrl/comparison_difftod.py` và config riêng `configs/comparison.difftod.json`.

Trạng thái hiện tại: code đã sẵn sàng cho smoke test, nhưng **chưa phải kết quả khoa học**.
`configs/difftod.models.lock.json` cố ý ở trạng thái `blocked_until_bootstrap`; cần bootstrap
revision bất biến, chạy H100 thật, chọn ngân sách matched trên dev, rồi khóa selection trước
khi đọc test.

## DiffTOD được giữ lại và phần nào được thích nghi

- Lõi planner vẫn là SEDD absorbing discrete diffusion, score-entropy loss, log-linear noise,
  analytic reverse sampler, semantic projection và MBR trên năm trajectory.
- Hai dataset dùng checkpoint fine-tune riêng. Cấu hình natural mặc định là SEDD-medium,
  sequence 1024, 100.000 optimization steps, 1.024 reverse steps và 5 candidates.
- Dữ liệu train có profile, target và toàn bộ trajectory chuẩn. Khi sinh plan đánh giá, condition
  không chứa reference dialogue và không chèn target trực tiếp; target chỉ đi vào bước sinh năm
  semantic guidance bằng Qwen local. Với DuRecDial, knowledge hợp lệ được dùng và condition quá
  dài được cắt token một cách xác định để vẫn nằm trong 1.024 token.
- DiffTOD sinh một immutable trajectory plan trước hội thoại. Qwen local chỉ hiện thực hóa plan
  theo lịch sử quan sát được; refusal guard và evaluator dùng lại nguyên hợp đồng của `crs_www`.
- Không có reinforcement learning trong baseline này. Đây là diffusion training có giám sát từ
  trajectory và score matching, sau đó sampling; PPDPP mới là baseline RL riêng.
- Evaluation khởi động đồng thời generation và embedding server giống runner hiện tại, dù DiffTOD
  không gọi embedding. Kết quả hạch toán cả Qwen tương tác, diffusion plan và phần guidance được
  phân bổ, đồng thời lưu số model calls/sequence forwards trong artifact.

## Các lớp bảo vệ tránh làm hỏng thí nghiệm hiện tại

- Không import code từ thư mục `../DiffTOD` lúc chạy và không gọi OpenAI/Anthropic.
- Mã tham chiếu DiffTOD được pin ở revision trong `configs/sources.lock.json`; implementation mới
  không sửa checkout tham chiếu đó.
- Asset inference chỉ được dùng khi revision và byte hash đã được khóa; inference đặt offline.
- Plan và guidance ghi atomically, không cho ghi đè artifact hoàn tất khác provenance.
- Test scenario không thể export, sample hoặc evaluate nếu chưa có selection lock hợp lệ.
- Natural và matched dùng namespace/path/hash riêng; một plan không thể bị dùng nhầm budget.
- Branch triển khai là `baseline`; không push tự động và không thay đổi branch gốc.

## 1. Chuẩn bị runtime và asset

Từ root `tdrl` trên login node của cụm H100:

```bash
uv sync --project runtime --locked --python 3.11

# Điền commit SHA 40 ký tự đã kiểm tra của hai Hugging Face snapshot.
runtime/.venv/bin/python scripts/difftod/bootstrap_assets.py \
  --sedd-revision SEDD_COMMIT_SHA \
  --tokenizer-revision GPT2_COMMIT_SHA \
  --output models/difftod \
  --lock configs/difftod.models.lock.json
```

Bootstrap là bước duy nhất cần mạng. Sau đó mọi job đặt `HF_HUB_OFFLINE=1` và
`TRANSFORMERS_OFFLINE=1`. Checkpoint base được tạo ở
`models/difftod/sedd-medium/SEDD_COMMIT_SHA/checkpoint.pth`; tokenizer nằm ở
`models/difftod/gpt2/GPT2_COMMIT_SHA/`.

Sau bootstrap, review rồi commit file lock đã có revision/hash thật trước scientific run. Script
sẽ từ chối ghi đè một lock đã ở trạng thái `pinned_and_verified`.

## 2. Chuẩn bị corpus train/dev

Input phải là canonical JSONL của `crs_www`, mỗi dòng có `id`, `profile`, `personality`,
`knowledge`, `target`, `messages`. Không dùng test để train hoặc chọn hyperparameter.

```bash
runtime/.venv/bin/python -m tdrl.baselines.difftod.cli prepare \
  --input data/processed/TopDial/train.jsonl \
  --output artifacts/difftod-data/TopDial/train.jsonl \
  --dataset TopDial --split train

runtime/.venv/bin/python -m tdrl.baselines.difftod.cli prepare \
  --input data/processed/TopDial/dev.jsonl \
  --output artifacts/difftod-data/TopDial/dev.jsonl \
  --dataset TopDial --split dev
```

Lặp lại với `DuRecDial2.0`. Nếu `data/processed/...` chưa tồn tại, phải chạy adapter dữ liệu
chính của `crs_www` trước; CLI DiffTOD sẽ dừng thay vì tự đoán schema.

## 3. Smoke test

Smoke CPU kiểm tra forward, backward, score-entropy và projected reverse sampling nhưng không
được coi là evidence khoa học:

```bash
runtime/.venv/bin/python -m tdrl.baselines.difftod.cli smoke \
  --output artifacts/difftod-smoke-cpu
```

Smoke H100 thật:

```bash
mkdir -p logs
sbatch --parsable scripts/slurm/difftod_smoke.sbatch
```

Job phải tạo `smoke.json` có `status=smoke_non_scientific`, `device=cuda` và thông tin H100.

## 4. Fine-tune hai checkpoint DiffTOD

Ví dụ TopDial; thay `SEDD_COMMIT_SHA` bằng revision đã khóa:

```bash
sbatch --parsable scripts/slurm/difftod_stage.sbatch \
  runtime/.venv/bin/python -m tdrl.baselines.difftod.cli train \
  --config configs/difftod/topdial.json \
  --train artifacts/difftod-data/TopDial/train.jsonl \
  --dev artifacts/difftod-data/TopDial/dev.jsonl \
  --tokenizer models/difftod/gpt2/GPT2_COMMIT_SHA \
  --base-checkpoint models/difftod/sedd-medium/SEDD_COMMIT_SHA/checkpoint.pth \
  --model-lock configs/difftod.models.lock.json \
  --output artifacts/difftod-train/TopDial
```

DuRecDial dùng `configs/difftod/durecdial.json` và output riêng. Có thể tiếp tục job bằng
`--resume artifacts/difftod-train/DATASET/checkpoint-last.pt`. Resume kiểm tra config hash và
khôi phục model, optimizer, EMA cùng RNG state.

## 5. Sinh guidance và natural plan trên dev

Trước tiên export đúng tập scenario dev đã chuẩn hóa:

File export chỉ giữ prefix đến user turn đầu tiên; các reference turn tương lai bị loại trước khi
guidance/sampling đọc file.

```bash
runtime/.venv/bin/python -m tdrl.baselines.difftod.cli export-scenarios \
  --root . --dataset TopDial --split dev \
  --output artifacts/difftod-scenarios/TopDial-dev.jsonl

sbatch --parsable scripts/slurm/difftod_stage.sbatch \
  runtime/.venv/bin/python -m tdrl.baselines.difftod.cli guidance \
  --root . --runtime-config configs/runtime.json \
  --scenarios artifacts/difftod-scenarios/TopDial-dev.jsonl \
  --output artifacts/difftod-guidance \
  --dataset TopDial --split dev --seeds 17 29 43
```

Sinh natural plans trên H100:

```bash
sbatch --parsable scripts/slurm/difftod_stage.sbatch \
  runtime/.venv/bin/python -m tdrl.baselines.difftod.cli sample \
  --config configs/difftod/topdial.json \
  --checkpoint artifacts/difftod-train/TopDial/checkpoint-final.pt \
  --base-checkpoint models/difftod/sedd-medium/SEDD_COMMIT_SHA/checkpoint.pth \
  --model-lock configs/difftod.models.lock.json \
  --tokenizer models/difftod/gpt2/GPT2_COMMIT_SHA \
  --scenarios artifacts/difftod-scenarios/TopDial-dev.jsonl \
  --guidance artifacts/difftod-guidance \
  --output artifacts/difftod-plans \
  --dataset TopDial --split dev --budget natural --seeds 17 29 43
```

Lặp lại hai lệnh guidance/sample cho DuRecDial2.0.

## 6. Chọn matched budget trên dev

Không hard-code matched config trước khi có timing H100. Tạo một lưới nhỏ các candidate, ví dụ:

```bash
runtime/.venv/bin/python scripts/difftod/make_matched_config.py \
  --base configs/difftod/topdial.json \
  --output artifacts/difftod-configs/topdial-128x1.json \
  --steps 128 --candidates 1
```

Chạy `sample --budget matched` cho từng candidate trên cùng dev scenarios và seeds, nhưng dùng
một `--output` riêng cho mỗi candidate để immutable plans không đè nhau. So sánh `wall_s`/plan
với budget H100 của phương pháp chính; chọn cấu hình gần nhất theo quy tắc đã định trước, không
theo test performance. Sau khi chọn, sinh lại matched plans vào plan root chính dùng cho evaluate.
File manifest của script luôn ghi
`candidate_not_selected`; cấu hình chỉ trở thành lựa chọn chính thức khi hash của nó được ghi vào
selection lock. Giữ candidate được chọn ở đường dẫn bất biến và dùng đường dẫn đó ở mọi lệnh sau.

## 7. Evaluate dev và khóa selection

Runner dùng cùng simulator, acceptance rule, seeds, protocols và turn budget của `crs_www`:

```bash
sbatch --parsable scripts/slurm/difftod_stage.sbatch \
  runtime/.venv/bin/python -m tdrl.baselines.difftod.evaluate_cli evaluate \
  --root . --output artifacts/comparison-difftod \
  --config configs/comparison.difftod.json \
  --plans artifacts/difftod-plans \
  --checkpoint-topdial artifacts/difftod-train/TopDial/checkpoint-final.pt \
  --checkpoint-durecdial artifacts/difftod-train/DuRecDial2.0/checkpoint-final.pt \
  --natural-config-topdial configs/difftod/topdial.json \
  --matched-config-topdial artifacts/difftod-configs/TOPDIAL_SELECTED.json \
  --natural-config-durecdial configs/difftod/durecdial.json \
  --matched-config-durecdial artifacts/difftod-configs/DUREC_SELECTED.json \
  --split dev
```

Sau khi kiểm tra report dev hoàn tất, khóa toàn bộ checkpoint/config/prompt:

```bash
runtime/.venv/bin/python -m tdrl.baselines.difftod.evaluate_cli lock-selection \
  --config configs/comparison.difftod.json \
  --checkpoint-topdial artifacts/difftod-train/TopDial/checkpoint-final.pt \
  --checkpoint-durecdial artifacts/difftod-train/DuRecDial2.0/checkpoint-final.pt \
  --natural-config-topdial configs/difftod/topdial.json \
  --matched-config-topdial artifacts/difftod-configs/TOPDIAL_SELECTED.json \
  --natural-config-durecdial configs/difftod/durecdial.json \
  --matched-config-durecdial artifacts/difftod-configs/DUREC_SELECTED.json \
  --development-evidence artifacts/comparison-difftod/dev/difftod-report.json \
  --base-selection artifacts/comparison/selection.lock.json \
  --output artifacts/comparison-difftod/selection.lock.json
```

`--base-selection` có thể bỏ nếu runner ba phương pháp chưa tạo lock; khi đó report bốn phương
pháp phải ghi rõ đây là selection độc lập.

## 8. Test khoa học

Chỉ sau bước 7 mới được chạy test. Dùng lại các lệnh export, guidance và sample nhưng thêm:

```text
--split test --selection-lock artifacts/comparison-difftod/selection.lock.json
```

Sau khi đã có cả natural và matched plans của hai dataset, chạy `evaluate` với `--split test` và
cùng `--selection-lock`. Runner sẽ từ chối nếu bất kỳ config/checkpoint hash nào khác selection.

## Kiểm tra trước khi commit hoặc gửi H100

```bash
python -m compileall -q src tests scripts/difftod
ruff check src/tdrl/baselines src/tdrl/comparison_difftod.py tests/test_difftod.py \
  scripts/difftod
pytest -q tests/test_difftod.py
git status --short --branch
```

Máy không có PyTorch/pytest chỉ có thể chạy compile/static audit; smoke tensor thật phải chạy ở
runtime đã khóa hoặc H100. Không được diễn giải compile pass là kết quả mô hình.
