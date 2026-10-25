# TopDial MRN debugging against DiffTOD

User authorization, 2026-10-12: retain Qwen/Qwen3.5-9B for this goal, rather
than the unresolved Qwen 3.8 model in the future comparison matrix. Embeddings
remain pinned Qwen/Qwen3-Embedding-8B, BF16, normalized 4096 dimensions. Focus
MRNFixed, root seed 17, profile/resistant protocols, allocation 58159. Preserve
all previous fitting, journals, locks and official-test results. Development
repairs cannot be selected from already opened official-test outcomes.

## Verified starting evidence

`artifacts/topdial-retrieval-candidate-debug-v6-58159/report.json` records 384
completed paired development dialogues: R successes 27/64 profile, 12/64
resistant; Full 26/64 profile, 13/64 resistant. Neither Full-minus-R paired CI
excludes zero. These are discovery diagnostics against fixed PPDPP, not DiffTOD
results or independent test confirmation.

`artifacts/mrn-rollout-terminal-audit-v1-58159/report.json` judges 70 prefixes
from 48 frozen MRN branches. One profile turn discarded all accepted branches;
one branch per protocol continued after acceptance; two resistant branches
continued after refusal. These hypothetical observations motivate a terminal
repair, not a causal explanation or measured success improvement.

## Opt-in terminal repair, v7

`PlannerConfig.semantic_rollout_terminals=true` requires revision 2 and the
existing frozen local semantic judge plus the current permitted scenario. After
each hypothetical user reply, judge that observed prefix with its actual finish
reason. Stop the branch after achievement, explicit target refusal or pressure.
Use zero remaining cost for achieved goals and exclude refusal/pressure branches.
Among nonterminal branches retain the selected MRN directed entry-plus-path cost.
Achieved branches precede nonterminal zero-cost ties; ties otherwise retain
candidate index order. If no branch is eligible, candidate zero remains the
explicit fallback; hypothetical refusal is not a real user refusal. Subsequent
real turns still use the unchanged shared outcome judge and refusal protocol.
Judge schema/transport errors fail closed; there is no regex fallback. Journal
IDs and derived judge seeds are distinct per candidate/depth. Record all observed
judgments, quoted evidence, judge protocol hash and actual calls/tokens. Default
false keeps the old branch execution. Semantic judgments concern simulated
continuations, not independent actual-user outcomes.

This changes Full's natural compute budget: at most six extra deterministic
judge calls for three branches/two lookahead steps, making the ceiling 18
Qwen calls per planning turn instead of 12. Early terminals may save lookahead
calls. A matched cap insufficient for the full declared branch budget dispatches
retrieval as before. Report the increased judge cost explicitly; do not claim
v7 has the same compute as v6. TDRL-R gains no terminal calls.

The frozen discovery config is `configs/topdial-semantic-terminal-debug-v7.json`.
Do not launch full evaluation with `run_topdial_debug.py` yet: that command
currently proceeds directly from smoke to evaluation. First run the worker
in smoke mode on each observed free GPU inside allocation 58159. Both GPU
runtime/controller/planner/journal replay gates and a fresh measured v7 feasibility
preflight are still required. Freeze the chosen candidate before opening the
existing disjoint 64-scenario development confirmation set. Reevaluate the fixed
DiffTOD adaptation on the identical confirmation scenarios and both protocols.

## DiffTOD fidelity audit and remaining implementation

Author source pin: `fd46b2887a9c0d9e0e413623e79c72ee32b9b7c1`,
<https://github.com/ninglab/DiffTOD/tree/fd46b2887a9c0d9e0e413623e79c72ee32b9b7c1>.
Inspected source code only; no upstream checkpoints, test dialogues, semantic
hints or historical metrics were imported.

- `load_model.py` loads the SEDD intrinsic diffusion policy and EMA checkpoint.
  Pin its architecture, tokenizer and initialization weights before fitting;
  inference-time weight downloads and broad loader fallback are inadmissible.
- `sampling.py:get_pc_sampler` applies conditioning projection before every
  diffusion step and final denoising; preserve analytic transition/denoising.
- `run_sample_topdial.py` uses GPT2 tokenization, 1024-token plans, 1024 analytic
  steps by default, profile tokens at the beginning and semantic guidance at
  position 192 shifted by profile length. Its test-resource entrypoint is not
  reusable. Generate guidance from permitted target/context using local Qwen,
  with provenance and measured calls; never read precomputed test guidance.
- `topdial_conversation.py:selectBest` performs self-BLEU consensus selection,
  excluding each sample's self score. Preserve planning and selection separately
  from shared Qwen realization. Its hosted generation/simulator/judge entrypoint
  must not be imported or executed.
- Training config defaults to PersonaChat, not TopDial; it specifies 100,000
  steps, AdamW, warmup 2500 and score-model EMA. Define a frozen TopDial-only
  train/dev adapter using this repository's audited splits, not bundled upstream
  datasets. Resolve source licensing and initialization provenance, training
  budget and BF16 feasibility before treating the baseline as valid.

DiffTOD remains **not implemented/fitted/evaluated** in this repository. A
prompt-only planning baseline cannot fill this slot. Outstanding work: isolated
source-faithful policy/sampler and train/dev adapter, forbidden-path audit,
fidelity fixtures, H100 update/replay and dual-residency smoke, method-specific
budget preflight, training/development selection and frozen paired evaluation.
No claim that TDRL-R or Full beats DiffTOD is supported yet.

## Source arithmetic fidelity, 2026-10-12

`src/tdrl/diffusion.py` now implements the native absorbing score-entropy
objective, log-linear noise, analytic reverse transitions, exponential-race
sampling, and profile/semantic-token projection at the original offset. It is
an intrinsic diffusion component, not yet a fitted or validated DiffTOD baseline.
`configs/difftod-reference.lock.json` pins hashes of the five inspected arithmetic
source files. `scripts/audit_difftod_arithmetic.py` requires those exact files,
imports only the source's pure arithmetic helpers, and never executes its dataset,
training or hosted-generation entrypoints.

The frozen `tests/fixtures/difftod-arithmetic-v1.json` records upstream CPU loss,
gradients, transition weights and a complete sampled token plan. Against the
source, loss/gradient agree within 1e-6, transition weights agree exactly, and
all five model-input trajectories plus the final sampled tokens agree exactly
for a four-step toy policy with seed 17. These are CPU mathematical/control-flow
checks only; there is no pretrained-policy execution or H100 fidelity claim.
Production natural-budget sampling still needs the native 1024 steps plus one
final denoising forward per plan. Conditioning overflow fails closed.

`configs/difftod-policy-model.lock.json` pins the native `louaaron/sedd-medium`
initialization and GPT2 tokenizer files, with immutable revisions, architecture
config and SHA-256 values. Anonymous bootstrap downloads are being checksum-
verified under `artifacts/difftod-policy-bootstrap`; this is not inference or
fitting. The diffusion transformer, checkpoint/EMA loader, train/dev adapter,
semantic-guidance generation, MBR consensus, lifecycle/compute integration and
all H100 policy gates remain pending. No upstream dataset or trained DiffTOD
checkpoint is reused.

## Runtime gate and prepared data artifacts

Both v7 smoke workers completed successfully in allocation 58159, steps 49/50,
with different GPU UUIDs, exact controller/plan/journal replay, zero optimizer
updates and zero unresolved requests. The output-hash audit is
`artifacts/topdial-difftod-goal/v7-smoke-audit.json`. These smokes use commit
41e3d98; later code changes do not silently inherit this execution gate. Slurm
records 231 and 224 allocated one-GPU seconds (455 H100-seconds total), including
cold loading/step overhead. Original smoke manifests' `h100_seconds=0` field is
incomplete; the separate Slurm evidence corrects segment accounting without
editing those immutable manifests. No v7 evaluation has run.

Data preparation command, from `tdrl/`:

```sh
.venv/bin/python scripts/prepare_difftod_data.py \
  --tokenizer artifacts/difftod-bootstrap-metadata/gpt2-tokenizer.json \
  --output artifacts/difftod-data-v1
```

`artifacts/difftod-data-v1/manifest.json` checksum
`95961aa999ec00519c2ad118c44b8f31c6050fdca6a2ae0d6efe95cd51574e8c`
records 12,601 audited train dialogues (8,169 length-1024 chunks) and 1,802
development dialogues (1,160 chunks), with ordered membership/content hashes,
tokenizer pin and per-1000-dialogue packing lineage. Native format uses profile,
target and observed dialogue; no outcome labels or future annotations enter the
training text. Keep non-recommendation goals. Each map group receives EOS tokens,
is concatenated separately and drops its final incomplete chunk; 5,122 train
and 604 development remainder tokens are disclosed. Group boundaries are fixed
at 1000 audited source rows rather than depending on datasets.map worker
partitioning. Development is only a separate held-out loss/checkpoint-selection
resource; its trajectories must never enter fitting or the TDRL graph.
Preparation supports exact completed-output reuse and durable split-boundary
resume. Changed content/membership/tokenizer/schema, unknown partial files, or
cross-split duplicate IDs fail closed. No official test bodies were opened.

## Checkpoint-compatible policy implementation

`src/tdrl/diffusion_model.py` implements native DDiT parameter names, adaptive
normalization, sigma embedding/scaling and RoPE. The tiny CPU fixture loads
upstream DDiT weights into this implementation and exercises nonzero modulation
and output gates: forward agrees within 1e-6; gradients within 2e-5. Source file
hashes are in `configs/difftod-model-reference.lock.json`; reproduce with
`scripts/audit_difftod_model.py`. Its CPU attention reference explicitly computes
matmul/softmax because native FlashAttention is unavailable on CPU. This proves
architecture/layout compatibility, not pretrained inference, H100 native-kernel
parity or scientifically fitted baseline fidelity.

Production loading requires one Slurm H100 before touching model weights,
checks every pinned file, rejects architecture/key/shape mismatch and never
uses remote-code loading or initialization fallback. Parameters publish BF16;
RoPE frequencies remain FP32 as auxiliary position arithmetic. PyTorch SDPA
replaces the upstream FlashAttention entrypoint. The original source uses FP32
parameters with mixed autocast, whereas this adaptation follows the project's
BF16 parameter contract; do not claim original numerical execution equality.
H100 forward/gradient/update and sampling replay plus actual dual residency
remain required before fitting or comparison.

`src/tdrl/diffusion_optimizer.py` uses native Torch AdamW math with explicit
learning-rate warmup, clipping and adaptive EMA decay
`min(ema_decay, (1+updates)/(10+updates))`. FP32 masters initialize from the exact
public weights rather than rounded BF16 parameters; moments and EMA retain FP32.
EMA averages the masters. Forward parameters are published in BF16 after each
update. CPU tests verify direct Torch AdamW/warmup/EMA equivalence, exact next-
update resume and rejection of nonfinite gradients or degraded saved precision.
This auxiliary-state execution is an explicit adaptation, not permission to
change the shared Qwen precision or backbone.

Anonymous bootstrap is resumable via `scripts/bootstrap_difftod_models.py`;
all files must match immutable SHA-256 before publication. Curl configuration
and credential-file lookup are disabled. Local download attempt 1 was truncated
at 17,602,292 bytes and rejected before any model load; its original partial and
hash are retained. A separate bootstrap on the SSH login host populates
`models/difftod`, with tokenization assets downloaded first. Bootstrap uses no
GPU or inference. Its manifest and process evidence distinguish running,
partial and complete state; a completed bootstrap still satisfies no H100 gate.

## Confirmation-safe loss validation

Do not use the previously prepared full `dev.npy` for DiffTOD checkpoint
selection: it includes the 64 frozen confirmation cases. The separate command
below verifies that frozen resource's population, IDs/order and selected content,
then excludes all confirmation provenance families before tokenization. This
keeps checkpoint selection from consulting the confirmation cases' offline
trajectories or their siblings. It reads no interactive outcome artifacts.

```sh
.venv/bin/python scripts/prepare_difftod_validation.py \
  --tokenizer artifacts/difftod-bootstrap-metadata/gpt2-tokenizer.json \
  --output artifacts/difftod-validation-v1
```

The resulting loss-validation corpus has 1,617 development dialogues / 1,041
length-1024 chunks, excluding 185 dialogues from confirmation families. Original
full dev preparation and immutable confirmation membership remain preserved.
Manifest SHA-256:
`af346fcb318071abfcb131937dbcb43713b26cab0ac1716e76e12114fb7c3323`
The manifest's outcomes flag refers to interactive simulation outcomes, not
raw corpus text used for checksum auditing. This is a declared dataset-selection adaptation; training
membership, fitting budget, both evaluation protocols and official test IDs
are unchanged. Freeze this rule before fitting, rather than choosing exclusions
from observed model success. Discovery/confirmation membership is disjoint in
scenario IDs, not a claim of independent user or training-seed replication.

## Real policy runtime smoke

Commit `9ef1d35` passed both allocated-GPU smokes with the public SEDD weights:
real BF16 training-chunk forward/gradient exact replay; eight-step policy
sampling exact replay and mask removal; resident BF16 Qwen generation and live
normalized 4096-D embedding/journal replay. Output hashes and absent pending
requests were audited in
`artifacts/topdial-difftod-goal/difftod-policy-smoke-audit.json`.
Slurm steps `58159.52` and `.53` completed with exit 0. Each monitor recorded
combined GPU memory, not just the controller allocator.

There were **zero optimizer updates**, and these eight-step static target-phrase
fixtures do not substitute for the native five-paraphrase/1024-step interactive
baseline. H100 update replay, native-kernel fidelity limitations, actual
interactive planning/MBR, method-specific full-budget preflight, fitting and
paired comparison remain pending. The SSH compute-node inspection attempt was
rejected before model launch; a read-only Slurm probe replaced it, with both
worker GPU telemetry retained. No unknown process was killed or reused.

## Interactive adapter and native-budget preflight

`src/tdrl/difftod.py` now produces five semantic target paraphrases with local
Qwen, applies native offset inpainting to five independently seeded 1024-step
SEDD trajectories, performs native self-BLEU consensus (NLTK 3.9.1, sentence
word tokenization, smoothing method 1, excluded self scores, first-index ties),
and realizes the selected hypothetical plan through the unchanged shared
assistant policy. It preserves native character-header/EOS plan extraction.
Hypothetical user choices/facts never enter observed dialogue history. Target
knowledge supplied to semantic paraphrasing is the permitted shared context;
this and the common realization/refusal protocol are disclosed adaptations.

The trajectory journal stores complete plans, token IDs, projection/seed traces,
policy provenance, guidance request metadata and checksums. Unknown pending
policy outcomes fail closed. A completed plan is reused on later real turns and
resume, and cached guidance tokens are attributed to the original first turn.
CPU control tests cover the five-branch budget, later-turn reuse, corruption
rejection and native header/EOS cases. The native consensus fixture is
`tests/fixtures/difftod-consensus-v1.json`. Source tokenization/selection agrees;
this still establishes no comparative dialogue success.

NLTK 3.9.1 and its added dependencies are locked without changing any existing
runtime package version. English Punkt resources are revision/hash-pinned in
`configs/difftod-nltk.lock.json`; bootstrap uses
`scripts/bootstrap_difftod_nltk.py`, with no downloads during inference.
Changing the package lock requires fresh runtime gates; older smokes retain
 their original source/lock provenance.

`configs/difftod-topdial-fit-v1.json` freezes the native effective 100,001-update
loop, batch 32, AdamW 1e-4, warmup 2500, gradient clip 1 and EMA .9999. Validation
uses the separate confirmation-family-safe corpus. The sampler's epoch/order/RNG
state and optimizer/EMA restore are serializable; validation temporarily
publishes EMA BF16 parameters and always restores online weights.

Before fitting, `scripts/preflight_difftod.py` requires actual gradient/update/
EMA replay, native batch-32 measurements, both full-budget profile/resistant
interactive dialogues and exact completed-plan/journal replay. It projects
fitting, validation, checkpoint I/O and all scoped method/budget dialogues
against allocation 58159's original clock with a safety factor. These unfitted
proxies cannot select a scientifically better planner or count as trained
DiffTOD evidence. Failure of the measured feasibility gate requires an explicit
budget/scope decision; do not silently shrink steps or evaluation membership.

The fitting runner `scripts/train_difftod.py` is gated by the audited native-
budget preflight and the original allocation clock. It records every update,
separate execution segments, durable optimizer/master/EMA/sampler/RNG state,
source-frequency snapshots and every improved EMA loss-selection candidate.
Resuming a durable checkpoint requires exact repeated next-update gradient/
optimizer/EMA replay before committing another update. Uncommitted log tails
remain preserved and are not counted as durable progress. Fitting and selected
policy manifests remain incomplete until all 100,001 updates and validation
selection finish. This runner does not open official test resources.
