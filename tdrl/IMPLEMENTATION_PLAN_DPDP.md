# DPDP-adapted implementation plan

## 1. Mục tiêu

Triển khai **DPDP-adapted** để so sánh công bằng với PPDPP, DiffTOD, TRIP,
TDRL-R và TDRL trên cùng experimental setting của `crs_www`:

- Datasets: TopDial và DuRecDial2.0.
- Official train/development/test splits.
- Seeds: `17`, `29`, `43`.
- Protocols: `profile`, `resistant`.
- Turn budget: `10`.
- Generation backbone: `Qwen/Qwen3.5-9B`, BF16.
- Embedding backbone dùng chung trong runtime: `Qwen/Qwen3-Embedding-8B`, BF16,
  4096 dimensions. DPDP không gọi embedding nếu cơ chế không cần.
- Cùng user simulator, assistant decoding, refusal guard và deterministic
  acceptance evaluator.
- Báo cáo cả natural budget và development-selected matched budget.
- Toàn bộ inference chạy local trong cùng Slurm allocation trên một H100 80 GB.

Đây là adaptation sang conversational recommendation, không phải exact reproduction
trên các domain ESConv, CIMA hay CraigslistBargain của paper gốc.

## 2. Ranh giới thay đổi

Mọi thay đổi chỉ được thực hiện trong:

```text
C:\study\Master_papers\code\CRS\crs_www\tdrl
```

và trên branch:

```text
baseline
```

Không chỉnh sửa:

```text
C:\study\Master_papers\code\CRS\DPDP
C:\study\Master_papers\code\CRS\DiffTOD
C:\study\Master_papers\code\CRS\TRIP

src/tdrl/baselines/difftod/
src/tdrl/baselines/trip/
code PPDPP hiện tại
code TDRL hiện tại
src/tdrl/comparison.py
```

DPDP sử dụng package và runner riêng. Trước commit phải kiểm tra
`git diff --name-status`; nếu có thay đổi ngoài phạm vi DPDP/config/docs/tests mới thì
dừng để rà soát.

## 3. Fidelity contract

### 3.1 Nguồn tham chiếu

```text
Repository: https://github.com/cs-holder/DPDP
Revision: 4d1e22e35db452d51f0f39388456d310ec1c971b
Paper: Planning Like Human: A Dual-process Framework for Dialogue Planning
```

### 3.2 Cơ chế bắt buộc phải giữ

- RoBERTa-large làm shared state encoder.
- Actor head dự đoán action distribution.
- Q-head dự đoán action values.
- Offline RL-based pretraining.
- MCTS-guided self-play training.
- Policy distribution làm MCTS prior.
- Open-loop MCTS.
- PUCT selection và tree value backup.
- Root action được chọn theo visit count.
- System 1/System 2 gate dựa trên chênh lệch xác suất top-1/top-2.

### 3.3 Nhánh thuật toán được chọn

TopDial và DuRecDial2.0 có complete trajectories, do đó adaptation dùng nhánh
complete-trajectory tương ứng ESConv/CraigslistBargain:

```text
complete trajectory
    -> per-turn reward
    -> discounted Monte Carlo return
    -> actor-critic offline pretraining
```

Không dùng cách huấn luyện fragment riêng của CIMA.

### 3.4 Discrepancy giữa paper và released code

Main implementation theo executable behavior của released code:

```text
advantage = return - Q(s, a)
actor_loss = -log pi(a | s) * stop_gradient(advantage)
critic_loss = MSE(Q(s, a), return)
```

Paper và code không hoàn toàn thống nhất về offline actor loss và self-play TD
target. Khác biệt này phải được ghi trong `fidelity.json`; không được âm thầm đổi
objective sau khi xem development hoặc test result.

## 4. Cấu trúc file dự kiến

```text
tdrl/
|-- README_DPDP.md
|-- configs/
|   |-- comparison.dpdp.json
|   |-- dpdp.models.lock.json
|   `-- dpdp/
|       |-- topdial.json
|       `-- durecdial.json
|-- scripts/
|   |-- dpdp/
|   |   |-- bootstrap_assets.py
|   |   `-- make_matched_config.py
|   `-- slurm/
|       |-- dpdp_smoke.sbatch
|       `-- dpdp_stage.sbatch
|-- src/tdrl/
|   |-- comparison_dpdp.py
|   `-- baselines/dpdp/
|       |-- __init__.py
|       |-- assets.py
|       |-- checkpoint.py
|       |-- cli.py
|       |-- config.py
|       |-- contracts.py
|       |-- dataset.py
|       |-- evaluate_cli.py
|       |-- mcts.py
|       |-- model.py
|       |-- planner.py
|       |-- rewards.py
|       |-- rollout.py
|       |-- selection.py
|       |-- serialization.py
|       `-- training.py
`-- tests/
    `-- test_dpdp.py
```

Không thêm dependency lớn mới. Implementation dùng PyTorch, Transformers, NumPy và
hạ tầng local Qwen hiện có. Không sử dụng FastChat cũ, OpenAI SDK, hosted Hugging Face
inference hoặc credential.

## 5. Config và typed contracts

### 5.1 Dataset config

Mỗi dataset có config riêng với các trường đã validate, ví dụ:

```json
{
  "dataset": "TopDial",
  "actions": [
    "acknowledge",
    "ask_preference",
    "clarify",
    "explain",
    "recommend",
    "farewell"
  ],
  "max_sequence_length": 512,
  "gamma": 0.999,
  "offline_epochs": 10,
  "offline_batch_size": 8,
  "offline_lr": 0.000006,
  "offline_critic_weight": 1.0,
  "selfplay_epochs": 3,
  "selfplay_cases_per_epoch": 100,
  "selfplay_lr": 0.000001,
  "selfplay_critic_weight": 1.0,
  "mcts_simulations": 10,
  "mcts_cpuct": 1.0,
  "mcts_q0": 0.0,
  "mcts_max_depth": 5,
  "mcts_max_realizations": 1,
  "critic_samples": 10
}
```

Config validation phải từ chối:

- Unknown field hoặc thiếu trường bắt buộc.
- Dataset không khớp config.
- Duplicate/unknown action.
- `mcts_simulations < 2`.
- Negative depth, batch size hoặc budget.
- Invalid decoding parameter.
- Model asset chưa được pin hoặc chưa được verify.

### 5.2 Typed contracts

Các kiểu dữ liệu chính:

```text
DPDPTransition
DPDPEpisode
PolicyOutput
TreeNodeStats
MCTSResult
RewardSample
DPDPCheckpointMetadata
DPDPPlanTrace
```

Mỗi transition phải lưu:

```text
state
action index
action name
reward
discounted return
next state
done
scenario ID
turn index
split
provenance hashes
```

## 6. Data adapter

### 6.1 Input

Đọc canonical train/development records của `crs_www`:

```text
id
messages
profile
personality
knowledge
target
assistant_actions
action_label_source
```

### 6.2 Tạo transition

Với mỗi assistant turn có user response tiếp theo:

```text
s_t   = observed history trước assistant turn
a_t   = assistant_actions tại turn đó
s_t+1 = history sau assistant và user response tiếp theo
done  = explicit acceptance hoặc end of trajectory
```

Không tạo transition từ assistant turn không có user response sau đó.

### 6.3 Policy input

Policy input dùng:

```text
[bounded target + profile header]
[recent complete observed dialogue turns]
```

Quy tắc:

1. Header có giới hạn token riêng.
2. Thêm complete turns từ mới đến cũ.
3. Không vượt quá 512 tokens.
4. Không chứa future turn, future action, success label hoặc future knowledge.
5. Cùng serialization được sử dụng ở training, inference và resume validation.

### 6.4 Output

```text
artifacts/dpdp-data/TopDial/train.jsonl
artifacts/dpdp-data/TopDial/dev.jsonl
artifacts/dpdp-data/DuRecDial2.0/train.jsonl
artifacts/dpdp-data/DuRecDial2.0/dev.jsonl
```

Mỗi output immutable và có manifest/hash.

### 6.5 Data tests

- Test split không thể được đọc trước selection.
- Không future-turn leakage.
- Action index và action order ổn định.
- State/next-state đúng thứ tự.
- Không duplicate transition ID.
- Không empty state hoặc unknown action.
- Không có test ID trong train/development artifact.
- Serialize rồi reload cho kết quả giống hệt.

## 7. Reward annotation

Reward annotation là bước H100 vì DPDP gốc dùng LLM critic.

### 7.1 Internal reward classes

```text
rejected    -> -1.0
no_progress -> -0.5
interested  ->  0.1
accepted    ->  1.0
```

Reward này chỉ dùng cho DPDP training và MCTS, không quyết định evaluation success.

### 7.2 Critic protocol

Với mỗi post-user state trong training split:

1. Gọi local Qwen critic 10 lần.
2. Mỗi request có deterministic derived seed và request ID riêng.
3. Output bắt buộc thuộc đúng một trong bốn labels.
4. Map label sang score và lấy mean.
5. Lưu raw outputs, labels, seeds, tokens, latency và prompt hash.

Nếu output malformed:

- Retry hữu hạn.
- Nếu vẫn lỗi thì stage fail.
- Không mặc định thành `no_progress`.
- Không bỏ transition một cách im lặng.

### 7.3 Discounted return

Tính ngược riêng trong từng dialogue:

```text
G_T = r_T
G_t = r_t + gamma * G_t+1
```

Không truyền return qua ranh giới hai dialogue.

### 7.4 Reward tests

- Label-to-score mapping.
- Discounted return khớp hand-calculated fixture.
- Terminal return đúng.
- Không nối returns giữa scenarios.
- Malformed critic output fail closed.
- Resume không gọi lại request đã hoàn tất.
- Không external inference.

## 8. Policy model

### 8.1 Architecture

```text
FacebookAI/roberta-large
    -> pooled output [B, 1024]
       |-- actor: Linear(1024, 6)
       `-- critic:
           Dropout
           Linear(1024, 1024)
           ReLU
           Dropout
           Linear(1024, 6)
```

DPDP và PPDPP có checkpoint hoàn toàn riêng dù cùng dùng pinned RoBERTa asset.

### 8.2 Forward contract

```text
Input:
  input_ids      [B, L]
  attention_mask [B, L]
  actions        [B]
  returns        [B]

Output:
  policy_logits  [B, 6]
  action_probs   [B, 6]
  q_values       [B, 6]
  actor_loss     scalar
  critic_loss    scalar
  total_loss     scalar
```

### 8.3 Loss

```text
q_selected = gather(Q(s), action)
advantage = return - q_selected

actor_loss =
    mean(-log_softmax(logits)[action] * stop_gradient(advantage))

critic_loss =
    mean((q_selected - return)^2)

total_loss =
    actor_loss + critic_weight * critic_loss
```

Numerical safeguards:

- Dùng `log_softmax`, không dùng `log(softmax())`.
- Assert finite logits, Q values, losses và gradients.
- Gradient clipping.
- Detach advantage trong actor loss.
- Không giữ một target Q-network không được executable path sử dụng.

### 8.4 Model tests

- Output shapes chính xác.
- Actor probabilities tổng bằng 1.
- Q gather đúng action.
- Loss khớp phép tính tay.
- Actor không backpropagate qua detached advantage.
- Critic và encoder nhận gradient.
- BF16 forward/backward finite.
- Save/load giữ nguyên logits và Q values.
- Action taxonomy mismatch khi load checkpoint phải fail.

## 9. Offline RL pretraining

```text
prepared transitions
    -> DataLoader
    -> RoBERTa actor/Q forward
    -> actor-critic loss
    -> optimizer + warmup scheduler
    -> checkpoint-last / checkpoint-best / checkpoint-final
```

Checkpoint selection chỉ dựa trên development evidence theo rule đã đóng băng.

Checkpoint phải lưu:

```text
model state
optimizer state
scheduler state
epoch
batch cursor
Python RNG
NumPy RNG
Torch CPU RNG
Torch CUDA RNG
config hash
data manifest hash
action taxonomy hash
model lock hash
source revision
```

Resume phải tiếp tục đúng batch kế tiếp, không train lại batch đã commit.

## 10. Open-loop MCTS

### 10.1 Tree identity

Mỗi actual decision tạo một tree mới. Node key là tuple các system actions tính từ
current root:

```text
()
("ask_preference",)
("ask_preference", "recommend")
```

Tree không được chia sẻ giữa scenarios hoặc actual turns.

### 10.2 Selection

```text
PUCT(s, a) =
    Q(s, a)
    + cpuct * prior(s, a) * sqrt(max(1, N(s))) / (1 + N(s, a))
```

Prior phải lấy từ RoBERTa policy checkpoint hiện tại.

### 10.3 Expansion và simulation

Mỗi expanded edge:

1. Qwen sinh assistant realization cho action.
2. Qwen internal user simulator sinh phản hồi.
3. Internal Qwen critic đánh giá state.
4. Realization được cache trong node.
5. Search tiếp tục đến terminal hoặc depth limit.

### 10.4 Terminal conditions

- Explicit acceptance.
- Explicit refusal.
- Search depth limit.
- Remaining actual turn budget bằng 0.

Sau explicit refusal không được tiếp tục tạo persuasion action.

### 10.5 Backup

Leaf value được backup bằng incremental mean cho toàn bộ traversed edges:

```text
N(s, a) <- N(s, a) + 1
Q(s, a) <- Q(s, a) + (value - Q(s, a)) / N(s, a)
```

### 10.6 Root decision

```text
chosen_action = argmax root_visit_count
```

Tie-break deterministic bằng frozen action order.

- System 1: tạo một Qwen realization.
- System 2: dùng best cached root realization từ MCTS.
- Cached internal user response không được đưa vào actual evaluation history.

### 10.7 MCTS tests

- PUCT khớp phép tính tay.
- Policy prior thật sự ảnh hưởng selection.
- Visit count tăng đúng.
- Q backup đúng incremental mean.
- Tree không bị chia sẻ giữa scenarios.
- Root action được chọn theo visit count, không theo raw Q.
- `mcts_simulations=1` bị từ chối.
- Depth limit hoạt động.
- Acceptance/refusal dừng rollout.
- Tie-break deterministic.
- Request IDs không trùng giữa branches.
- Same seed tạo cùng tree trace.

## 11. MCTS-guided self-play training

Mỗi training episode:

```text
training scenario background
    -> MCTS chọn action
    -> assistant realization
    -> internal user simulation
    -> critic reward
    -> lưu log pi(a|s), Q(s,a), reward
    -> tính discounted episode returns
    -> actor-critic update
```

Trong self-play, mọi executed action do MCTS chọn. Policy học từ trajectory thực sự
được thực thi; không train trực tiếp từ toàn bộ transition nội bộ trong tree nếu released
execution path không làm vậy.

```text
returns = discounted_rewards(rewards)
advantages = returns - Q_selected

L_actor = -mean(log_prob * stop_gradient(advantages))
L_critic = MSE(Q_selected, returns)
```

Self-play chỉ sampling background từ training split. Development chỉ dùng cho selection;
test vẫn sealed.

## 12. Inference planner

```text
Observed state
    -> RoBERTa policy probabilities + Q values
    -> confidence_gap = top1 - top2
    -> compare with frozen eta

gap >= eta                         gap < eta
    -> System 1                       -> System 2
    -> policy argmax                  -> MCTS
              \                     /
               -> final response
               -> common evaluation user simulator
               -> common deterministic acceptance classifier
```

Actual user response luôn do common `run_dialogue()` tạo. DPDP rollout simulator chỉ
tồn tại bên trong MCTS tree.

## 13. Control-gate threshold

Không cập nhật threshold theo test stream như source code gốc.

Quy trình:

1. Thu confidence gaps trên fixed development scenarios.
2. Thử các ratio/threshold candidates được khai báo trước.
3. Đánh giá development performance và compute.
4. Chọn threshold theo frozen selection rule.
5. Ghi threshold vào selection lock.
6. Test chỉ đọc threshold đã khóa.

Artifact phải lưu:

```text
threshold
development gaps
actual MCTS participation rate
policy-turn count
MCTS-turn count
config hash
checkpoint hash
development scenario IDs
```

## 14. Natural và matched budgets

### 14.1 Natural budget

Giữ gần released method:

```text
MCTS simulations: 10
max realizations per action path: 1
cpuct: 1.0
natural search depth
critic samples per state: 10
development-selected frozen gate threshold
```

### 14.2 Matched budget

Trên development, thử grid được khai báo trước:

```text
simulations: [2, 4, 6]
depth:       [1, 2, 3]
```

Matched selection chỉ được thay đổi decoding/search compute. Không được thay:

- Training checkpoint.
- Reward definition.
- Qwen model.
- Action taxonomy.
- Training objective.
- Training data.

Chọn candidate có measured H100-seconds/dialogue gần target nhất, không chọn theo test
performance. Natural và matched phải có config hash, namespace và output path riêng.

## 15. Evaluation runner

Tạo runner riêng:

```text
src/tdrl/comparison_dpdp.py
configs/comparison.dpdp.json
```

Runner mở rộng scope thành:

```text
PPDPP
DiffTOD
TRIP
DPDP
TDRL-R
TDRL
```

DPDP dùng lại common:

- Scenario membership và order.
- Seeds `17/29/43`.
- `profile` và `resistant` protocols.
- Turn budget `10`.
- User-simulator prompt.
- Deterministic acceptance evaluator.
- Refusal guard.
- Assistant/simulator decoding.
- Metrics và paired bootstrap.

Mỗi actual turn ghi:

```text
policy probabilities
Q values
confidence gap
frozen threshold
selected system
selected action
MCTS simulations
expanded nodes
search depth
visit counts
tree Q estimates
generation requests
input/output tokens
latency
H100 seconds
checkpoint/config hashes
```

## 16. Selection lock và test sealing

Selection lock phải chứa:

```text
offline checkpoint hash
self-play checkpoint hash
RoBERTa asset hash
action taxonomy hash
reward prompt hash
rollout prompt hashes
MCTS natural config
MCTS matched config
gate threshold
development scenario IDs
development evidence
Qwen model locks
comparison config hash
code commit
```

Test command phải từ chối chạy nếu:

- Git state không khớp selection.
- Checkpoint/config/prompt/action order thay đổi.
- Model revision hoặc weight hash thay đổi.
- Natural/matched config thay đổi.
- Development evidence thiếu.
- Selection lock không hợp lệ.

## 17. CPU test plan

CPU tests dùng tiny/fake components, không phải scientific evidence.

```bash
python -m compileall -q src tests scripts/dpdp
ruff check src/tdrl/baselines/dpdp src/tdrl/comparison_dpdp.py tests/test_dpdp.py scripts/dpdp
pytest -q tests/test_dpdp.py
pytest -q
```

CPU smoke phải hoàn thành:

```text
tiny policy forward
offline loss backward
tiny MCTS search
one self-play update
checkpoint save/resume
one fake interactive turn
manifest generation
```

Toàn bộ existing tests của TDRL, PPDPP, DiffTOD và TRIP phải tiếp tục pass.

## 18. H100 smoke plan

### Gate A — Policy model

- Load pinned RoBERTa-large.
- BF16 forward và backward.
- Actor/critic losses và gradients đều finite.
- Save/load checkpoint.
- Ghi peak allocated/reserved VRAM.

### Gate B — Live MCTS

- Khởi động Qwen generation và embedding servers.
- Giữ RoBERTa policy trên cùng H100.
- Chạy một MCTS decision với 2 simulations.
- Xác minh policy prior, expansion, critic, backup và visit-count selection.
- Ghi calls, tokens, latency và VRAM.
- Xác minh không có external inference.

### Gate C — Interactive dialogue

- Chạy một actual assistant turn.
- Chạy một common user-simulator turn.
- Chạy common acceptance classification.
- Xác minh internal rollout user không đi vào actual history.
- Xác minh refusal guard.

### Gate D — Training và resume

- Chạy một offline batch.
- Chạy một self-play episode.
- Interrupt sau checkpoint.
- Resume đúng batch/episode tiếp theo.
- Kiểm tra action, log-probability và gradient replay.

Nếu bất kỳ gate nào fail thì không chạy full training.

## 19. Failure-prevention checklist

| Nguy cơ | Cách chặn |
|---|---|
| Sai action index | Taxonomy hash trong data và checkpoint |
| Future/test leakage | Prefix-boundary và split-sealing tests |
| MCTS tree dùng nhầm scenario | New tree cho mỗi actual turn |
| Root không có visit | Validate simulations tối thiểu bằng 2 |
| NaN/Inf trong loss | Finite assertions trước optimizer step |
| `log(0)` | Dùng `log_softmax` |
| Critic output malformed | Strict enum parser, bounded retry, fail closed |
| Resume lặp batch | Exact cursor và toàn bộ RNG states |
| Request ID collision | Hierarchical deterministic namespace |
| MCTS user trở thành actual user | Tách internal và evaluation interfaces |
| DPDP tự chấm success | Common evaluator quyết định metric |
| Threshold thay đổi trên test | Frozen development threshold |
| Hosted inference lọt vào | Static import/endpoint/credential audit |
| Artifact bị ghi đè | Atomic write và immutable output directory |
| Nhầm natural/matched | Budget nằm trong namespace và manifest |
| Regression mô hình khác | Full existing test suite và diff-scope audit |

## 20. Trình tự triển khai và commit

Các commit nhỏ, dễ kiểm tra và rollback:

1. `Add isolated DPDP contracts and data adapter`
2. `Implement DPDP policy and offline RL training`
3. `Implement policy-prior open-loop MCTS`
4. `Add DPDP MCTS-guided self-play training`
5. `Add DPDP evaluation and budget selection`
6. `Add DPDP H100 scripts documentation and tests`
7. `Fix issues found by full regression smoke`

Tất cả commit chỉ nằm trên branch `baseline`. Không push hoặc merge nếu chưa được yêu cầu.

## 21. Definition of done

Implementation chỉ được xem là hoàn thành khi:

- DPDP upstream repository không thay đổi.
- Không sửa implementation của PPDPP, DiffTOD, TRIP hoặc TDRL.
- Compile pass.
- Ruff pass.
- DPDP unit tests pass.
- Toàn bộ existing test suite pass.
- CPU smoke pass.
- Data leakage và split sealing tests pass.
- Offline loss fidelity tests pass.
- MCTS selection/backup tests pass.
- Checkpoint/resume tests pass.
- Hosted-inference audit pass.
- README có đầy đủ lệnh bootstrap, prepare, reward annotation, train, smoke,
  development selection, lock và test.
- Natural/matched configs được tách rõ.
- `git diff` chỉ chứa các file DPDP và integration mới đã cho phép.
- Commit thành công trên branch `baseline`.
- Working tree sạch sau commit.

H100 scientific readiness chỉ được xác nhận sau live smoke. CPU smoke, compile pass và
unit tests không được diễn giải thành kết quả khoa học hoặc bằng chứng H100.
