# Fixed 500 TopDial + 500 DuRecDial evaluation

User amendments (2026-10-14): freeze and publish explicit IDs across devices,
run 200 scenarios per dataset first, then support extending to 500 per dataset.
This creates a new 1000-case resource; the original pooled seed-42 500-case
resource and its completed reports remain unchanged.

`src/tdrl/resources/test_scenarios_1000_td500_dr500_v1.json` contains explicit
IDs, zero-based per-dataset indices, order, original content hashes, population
hash, and immutable provenance. Retain the original 330 TopDial and 170 DuRecDial
IDs in their original order. Append 170 TopDial and 330 DuRecDial IDs chosen once
from the remaining sorted population with `random.Random(42)` per dataset.
This is a stratified extension, not a new pooled-uniform 1000-case sample.
The runtime never samples and rejects changes rather than choosing substitutes.

Execution order: 200 TopDial, 200 DuRecDial, remaining 300 TopDial, remaining
300 DuRecDial. `--count 400` thus means the first 200 cases of each dataset;
`--count 1000` means the final 500 of each. Both methods run natural budgets,
profile/resistant protocols and experimental root seed 17. The initial comparison
has 1600 dialogues; the extended comparison has 4000 dialogues.

```sh
PYTHONPATH=src runtime/.venv/bin/python scripts/run_fixed1000_prefix.py \
  --count 400 --output artifacts/fixed1000-prefix-tdrl-difftod-v1-58159
# Extend with the same source, output root and immutable selection:
PYTHONPATH=src runtime/.venv/bin/python scripts/run_fixed1000_prefix.py \
  --count 1000 --output artifacts/fixed1000-prefix-tdrl-difftod-v1-58159
```

Launch the queue as a detached process in a clean committed checkout. It waits
for the current independent TopDial development confirmation and idle owned
allocation 58159. It prepares DuRecDial train/dev diffusion data, runs fresh
native gradient/update/interactive replay preflights on both H100s, and fits a
separate DuRecDial diffusion policy for exactly 100001 updates. Retain native
SEDD likelihood, five semantic paraphrases, 1024-step analytic sampling and
self-BLEU consensus; select minimum development-loss EMA. Dataset entrypoints share
the native training/preflight/packing implementations; policy, optimizer, model
and replay code remain unchanged. Only dataset/preparation/stage paths differ.
Train/dev only; no new official test outcomes are used for fitting or selection.

TopDial uses the v12 MRNFixed Full candidate frozen before independent confirmation.
DuRecDial uses the original globally selected IQE Full with original thresholds,
training-built bank and planner revision 1. This is explicitly a retained DuRecDial
method version, not a DuRecDial v12/MRN refit. Shared Qwen3.5-9B, Qwen3-Embedding-8B,
BF16, simulator, semantic judge, decoding, turn limit and seeds remain fixed.

Before test bodies are opened, freeze a new joint selection lock with both
checkpoints, head/bank hashes, official archive hashes, shared model/config/package
pins, original selection, confirmation report and this committed runner. Then run
fresh real dialogue/replay smokes for every dataset/method on both GPUs and a
measured feasibility gate using the original allocation clock. A failed runtime
or budget gate stops the queue with durable evidence; no hidden fallback.

Population metadata extraction for the new ID resource read only IDs from the
hash-verified release under the already verified original selection lock. No
conversation, target, knowledge or outcome was exposed or used in membership.
At the first post-selection test opening, freeze normalized content hashes for
all 1000 IDs in `sealed-sample.json`; known original hashes are checked immediately.
All official archives and adapter/source code are pinned, making new content
reproducible on another device before that seal is copied. IDs, order, indices,
populations, archives, checkpoints or content mismatches fail closed.

Stopping count and GPU slot do not enter dialogue request IDs. Completed snapshots
carry immutable checksum seals and are reused on extension. Interrupted dialogues
resume with their saved history and durable request journals. Per-prefix manifests,
metrics and paired reports remain separate. Use the same source, selection, HPS
output and journals for `400 -> 1000`; do not overwrite previous evidence. Download
large data/checkpoints separately: GitHub contains IDs, code and config, not private
traces or model weights. All experiment outputs live under `/mnt/data/duclm3`.

Single-seed local interactive simulation on sampled official tests; not full
splits, training-seed uncertainty or human-study completion. Report paired 95% CIs
with 10000 scenario-cluster bootstrap resamples and measured compute. Queued or
submitted work is not completed evidence.
