# Proposed single-allocation experiment protocol

Status: **design only; not measured feasibility and not experimental results**.
TDRL means Temporal Distance Reinforcement Learning provisionally.
All scientific GPU stages belong to one newly submitted Slurm job on SSH alias `H100`,
one node, one full H100 80 GB, BF16, wall limit `20-00:00:00` (480 hours). No nested `sbatch`,
second GPU, quantization, model substitution, or CPU model fallback is permitted.

## Live cluster inspection, 2026-10-03

Read-only SSH succeeded. Account association is `normal`, QOS `normal`; no user jobs were
present at inspection. Partition `main` advertises unlimited MaxTime and full H100 80 GB GRES
`gpu:nvidia_h100_80gb_hbm3`. `worker-0` was mixed with 5/8 GPUs allocated; other main nodes were
drained. These are scheduling observations, not a reservation or guarantee of 20-day access.
Inspect `squeue -u "$USER"`, `sinfo -N -p main`, association/QOS limits immediately before
submission. Driver/VRAM evidence must come from the allocated worker; login-host NVML failed.
Remote default Python is 3.10.12, so bootstrap a separate Python 3.11 environment with uv.

## Before requesting the allocation

1. Implement and validate official adapters against source documentation and small training
   fixtures; audit licenses, official split membership and train/dev leakage. Never inspect
   test dialogue bodies before the selection lock. Source metadata and archive topology may
   be inspected without loading test dialogues. Preserve the original immutable raw files.
2. Bootstrap the pinned local runtime and download both pinned model snapshots, plus permitted
   intrinsic baseline policy resources. Verify actual weight bytes and tokenizer/template hashes.
   Pin those additional resources separately. Do not download or run original hosted client code.
3. Complete CPU control tests and stage runners, scenario ordering, all prompt templates,
   deterministic acceptance protocol, resistant-user protocol, and baseline adaptation tests.
   The current implementation does **not** meet all these prerequisites; see STATUS.md.
4. Commit the new implementation and record dirty state/config/lock hashes. Freeze the expected
   stage output topology, resumable journaling contract, and full workload inventory.
5. Obtain institutional determination before human recruitment or labeling. GPU simulation
   work may precede approved human labeling, but Gate 6 remains pending until that study finishes.

## Allocation schedule and stop conditions

The following are **maximum planning envelopes**, not measured runtime predictions. Their sum
is 480 hours. Bootstrap is outside this schedule. Queue time is outside the allocation.

| Stage | Hours | Required exit evidence |
|---|---:|---|
| Gate 2 and warm profiling | 6 | Dual BF16 residency, full live-embedding dialogue, both dispatch branches, exact resume/replay, VRAM/network/compute manifests |
| Offline training-state embeddings | 12 | Full 4096-D normalized training bank with exact provenance/order and resumable chunks |
| MRNFixed/IQE learning | 18 | Equal data/optimizer/step budgets; detached Bellman and separation tests; all checkpoints retained |
| Baseline policy fitting | 48 | Four faithful adaptations trained solely from permitted training resources, provenance and fidelity tests |
| Development calibration/selection | 48 | Training-built maps; separate H_edge/H_cover curves; paired SR/compute selection; frozen adaptations/prompts/budgets; immutable lock |
| Primary test, both user protocols | 240 | Full official splits, all methods, natural/matched settings, three seeds, paired valid scenario sets |
| Coverage/transfer/mechanism diagnostics | 36 | Independent labels, holdouts, both transfer directions, representation/routing/recovery analyses |
| Reporting and human packet export | 6 | Validated completion inventory, generated tables/figures; blinded study packet only if approved |
| Contingency and final drain | 66 | Includes at least 24 hours reserved for failures/checkpoint/export; no silent scope reduction |
| Total | **480** | Submission remains capped at 20 days |

Serve generation and embeddings together only during interactive stages. Offline embedding
may use the embedding model alone. Shut down only this job's owned server PIDs before head or
intrinsic policy training; use the same GPU sequentially. Restart both for interactive
development/test, retaining cold-load measurements. Do not leave policy training and servers
contending for memory. Intrinsic baseline policies must also fit alongside both Qwen models at
evaluation; residency cannot be assumed from the two-model smoke alone.

Initial vLLM settings reserve 52% for generation and 40% for embeddings, context 4096, eager
execution, one tensor-parallel GPU, 4/8 simultaneous sequences. These are **unvalidated startup
settings**, not measured VRAM. Startup is sequential and loopback-bound. Reduce KV/concurrency
within the declared development/runtime budget if needed; do not change precision, model, GPU
count, or model placement. If dual residency fails, preserve logs/memory observations and stop.
Context overflow rejects the request; corpus truncation requires a documented preprocessing
decision and hashes, never silent server truncation.

## Workload and feasibility calculation

Required methods are Qwen without planning, TDRL-R, full TDRL, cosine control, random routing,
always-rollout, **PPDPP, DPDP, TRIP, DiffTOD**. All realization, LLM planning, simulation, and
diagnostic moderation use the same locked Qwen/Qwen3.5-9B. Cross-backbone comparisons are excluded.
Intrinsic policy models remain method-specific; language generation is unified.

Run natural settings for all ten methods. Also run matched settings for the eight planning
methods other than Qwen-only/TDRL-R. If development establishes that a method's natural and
matched configuration is identical, use one immutable trace set with explicit alias metadata;
never call copied traces independent trials. The conservative capacity plan reserves all 18
settings before such deduplication. Both profile-conditioned and frozen resistant-user
protocols, seeds 17/29/43, and every official test scenario are required.

Let N be the **audited sum of official test scenario counts across both datasets and all their
test splits**. The primary workload reserves `18 × 2 × 3 × N = 108N` dialogues. The provisional
CLI topology gives 324 configuration cells (TopDial seen/unseen, DuRecDial test), but is explicitly
unverified and must be rebuilt from official membership; it is not a dataset-count claim.

At runtime, measure warm p50/p95 complete-dialogue seconds for every method/budget/protocol/
dataset, actual calls/tokens/turns, cold loading, and observed full-stage progress. Profile on
development only; no test feedback enters configuration selection. Evaluate feasibility twice:
after Gate 2 with provisional timings, and after development with the actual full matrix.

```
estimated_test_hours = 1.25 * sum(N_dataset_split * measured_dialogue_p95_cell) / 3600
available_hours = remaining_allocation_hours - reserved_diagnostics_export_failure_hours
```

`tdrl budget` consumes those measured values; it fails when counts/timings are absent. Its
default 24-hour reserve is only the final minimum: the orchestrator must subtract all remaining
non-test stages before passing remaining hours. The 25% factor is a declared conservative
estimate, not a confidence guarantee. If the full workload does not fit, emit a budget failure
artifact and stop before test, asking for an explicit change. Never drop a dataset, baseline,
protocol, seed, human study, or difficult scenario to make the deadline.

## Training, graphs, selection, and independent support diagnostics

Both heads share observed-history serialization, normalized full-size Qwen embeddings, sampled
training transitions/goals, candidate seeds, optimizer, 30,000 proposed optimizer steps per
candidate, minibatch 256, learning rate 3e-4, positive c_step=1, gamma=1, EMA tau=.01. These
numbers are proposed development settings. Any change must be shared across head candidates
and recorded before fitting. Frozen embedding model weights are never updated.

Sample reachable goals from the same legitimate training trajectory after the source; include
goal/self and one-step predecessor targets. Failed terminals/censored pairs do not receive a
reachable Bellman mask; use a separately declared negative margin objective. Keep the target
head detached. Retain optimizer, target-network, RNG/sampler state and checkpoint signatures.
Complete the dataset-aware sampler before training; the current learner supplies the objective,
not source-grounded reachability labels.

Build all references/edges/guidance from training. Bound graph construction with chunked scoring;
do not silently substitute cosine neighbors for directed connectivity. Measure bank size and
edge density on training before the 12/18-hour envelope is accepted. Cosine is a symmetric
control; it is not a directed quasimetric. Keep d(source,target) argument order in every stage.
Calibrate connectivity H_edge and dispatch H_cover independently per dataset on development.
Select MRNFixed/IQE by the AGENTS.md paired SR/compute/parameter rule and freeze once. Test
commands must verify selection.lock and matching actual code/config/prompt/model contracts.

Create applicability labels without learned distances, controlled trajectory/domain holdouts,
next-move/strategy compatibility, temporal order and remaining-step diagnostics. Freeze the
holdout rule before fitting and use only allowed train/dev selection resources. Test annotations
do not retrain or calibrate. Compare lexical/generic semantic/cosine/MRNFixed/IQE on the same
labels; report AUROC/AUPRC and prevalence. Coverage means retrieval-defined support; pre-action
failure prediction is not intervention-invariant. Transfer both ways with source-development
thresholds and no destination test calibration. Qualitative case selection uses a fixed hash-
ordered rule for covered/off-map/recovery-success/recovery-failure, including absent categories.

## Fair evaluation, success, autonomy, and compute

All methods receive the same allowed profile/knowledge/target information, conversation history,
scenario order, turn budget (proposed 10), paired seeds, final-length cap, user protocol, refusal
guard, and deterministic acceptance protocol. Hidden dataset outcomes/future turns are excluded.
Only explicit, unpressured target acceptance counts; mention/recommendation/rejection remain
distinct. **Dataset-grounded acceptance implementation is pending.** Do not equate a target
string match, simulator judge, or mention with acceptance. Qwen moderator disagreements remain
separate diagnostics.

Calibrate matched budgets on development using actual assistant/planning/simulator calls,
generated tokens, and attributed H100 seconds, reserving capacity for the final response.
Freeze candidate/depth/pruning caps and budget-exhaustion behavior for every planner. Report
remaining actual resource differences even when nominal caps match. Random routing has a
development-frozen trigger probability and budget caps; always-rollout is an upper-compute
control. Do not stop evaluation when a method performs badly or overspends; account for its
declared budget policy consistently.

Log every request by role with seed/template/prompt/model hashes, token counts, latency and
retries. Attribute sequential dialogue wall time to its one resident H100; retain idle overhead,
cold load and stage training time separately, plus total allocation GPU seconds. Avoid double-
counting overlapping requests. Report generation/embedding calls per turn/dialogue, embedding
texts/tokens, warm p50/p95, peak VRAM, trigger/candidate/lookahead counts, errors/retries and
H100-seconds per dialogue/success. A failed dialogue counts the full turn budget even when a
refusal guard ends persuasion early; actual compute remains actual measured compute.

Infrastructure failures are not policy failure labels. Preserve them and rerun the affected
paired methods on the same valid scenario set inside this allocation. Unresolved errors or
missing scenarios leave status partial. Never claim completion from partial shards. Report
per-seed and aggregate SR, average turns, compute, frozen coverage strata and recovery metrics.
Use 10,000 scenario-cluster paired bootstrap resamples, fixed seed 20261003, preserving seeds
inside each sampled scenario and stratification across datasets in macro comparisons.

The refusal guard is sticky and conservative; its English/Chinese explicit forms need audit.
Track naturalness, helpfulness, coherence, appropriateness, repeated persuasion, autonomy and
refusal violations alongside SR. The guard is applied to **all** methods. Simulation remains
same-family interactive evidence, not a live-user deployment claim.

## Human study and paper gate

After institutional determination: 100 fixed test scenarios per dataset, TDRL vs TDRL-R and
TDRL vs the strongest development-selected non-TDRL baseline, three independent raters/pair,
independent blind A/B randomization, ties permitted. This is 400 pairs / 1,200 pair-level ratings.
Hide method names/metrics/traces and outcomes not visible in the dialogue. Record recruitment,
locale/inclusion criteria, compensation/hourly rate, consent, quality controls, anonymization,
data handling and determination ID. Report item-level anonymized judgments, win/tie/loss,
paired uncertainty and agreement. External approvals/raters are not promised within GPU wall
time; do not mark Gate 6 completed before they finish.

Generate all tables/figures from new manifest-backed outputs, with run IDs/paths in source
metadata. Build and inspect the paper with latexmk only when claims are backed by artifacts.
Historical PDFs/repositories inform design/reviewer concerns only and supply no numbers.
