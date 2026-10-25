Training semantic strategy overlay
=================================

The frozen 32-move training diagnostic found 29 `recommend`, two `explain`, and one `ask_preference` semantic labels among target-mentioned question moves weak-labelled `ask_preference`. That intentionally biased subset supports a guidance-error hypothesis; it is not a population label-error rate or success estimate. Artifact: `artifacts/topdial-strategy-audit-v1-58159/report.json`, source commit `452f584`.

The new annotation covers all 58,364 actionable states in the original training-built TopDial bank. The frozen classifier sees the declared training target, preceding observed history, and next observed assistant move. It does not see weak labels, distances, future users or outcomes. The prompt is unchanged from the diagnostic. Generation and embeddings remain the pinned local BF16 Qwen models on one H100 in allocation 58159.

From a clean isolated checkout, inside an exclusive one-GPU/eight-CPU Slurm step with the existing locked offline runtime:

```sh
runtime/.venv/bin/python scripts/annotate_training_strategies.py --slot 1 --mode preflight --output artifacts/topdial-semantic-strategy-v1-58159/preflight
runtime/.venv/bin/python scripts/annotate_training_strategies.py --slot 1 --mode annotate --gate artifacts/topdial-semantic-strategy-v1-58159/preflight --output artifacts/topdial-semantic-strategy-v1-58159/annotation
```

Use `TDRL_POLICY_DETERMINISM=strict-backward-v1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; the owned vLLM launcher removes controller-only settings from server workers. An isolated worktree needs verified links to the existing model cache, processed training data, toolchain and runtime. Set `PYTHONPATH` to its `src` and `scripts`, because the shared editable runtime otherwise resolves the original checkout. Do not change the source tree of an active fitting worker.

Preflight labels 128 training nodes sorted by `digest([17,node_id])`, verifies real live 4096-dimensional embedding and exact completed-request replay, and projects total annotation using the slowest measured request plus cold loading. It retains the conservative original DiffTOD comparison reserve and original allocation clock. A failed runtime or feasibility gate blocks full annotation; post-fit method-specific comparison feasibility remains required.

Request journals persist labels across interruption with exact code/config/bank/model/prompt/seed contracts. Unknown, truncated or unresolved outcomes fail closed. `stop.requested` stops after a completed request. Partial outputs are not accepted as an overlay. Prior failure manifests and completed request evidence remain available; unknown segment compute must not be reported as measured cumulative compute.

`apply_strategy_overlay` returns copies of original bank records; it changes only their training action and excludes ambiguous guidance. It retains the bank order, state serialization, embeddings, targets and head. Original bank records and labels remain immutable. `configs/topdial-semantic-strategy-debug-v8.json` enables the completed hash-verified overlay in fresh MRNFixed TDRL-R/Full development diagnostics. Run the existing both-GPU planner runtime/replay gates before evaluating this configuration. Discovery results select a candidate; disjoint confirmation and a fixed fitted DiffTOD baseline are still needed to establish any improvement.

After a complete paired discovery report, freeze a candidate only if both MRN methods have positive macro-average development differences versus fixed DiffTOD:

```sh
runtime/.venv/bin/python scripts/freeze_topdial_confirmation.py --report artifacts/topdial-semantic-strategy-debug-v8-58159/paired-difftod/report.json --candidate-config configs/topdial-semantic-strategy-debug-v8.json --output artifacts/topdial-semantic-strategy-debug-v8-58159/confirmation-candidate.lock
```

This command recomputes the paired report from its three hashed complete source stages. It checks current code/config/package provenance, identical MRN head and separately calibrated thresholds, the original completed strategy overlay, fixed fully fitted DiffTOD and disjoint frozen membership. The write-once lock pins these values. It does not run confirmation or open outcomes; confirmation runners must verify and use this lock before execution. A positive discovery point estimate is a selection prerequisite, not proof of superiority.

A training-only anchor diagnostic on the same frozen 128-node preflight found 80 target-mention anchors: 63 semantic recommendations, 10 explanations, two preference questions and five farewells. Inspection of all five farewell examples shows preceding users thanking/accepting or ending the conversation. Artifact: `artifacts/topdial-semantic-strategy-v1-58159/preflight-goal-anchor-diagnostic.json`. These counts are a sample diagnostic, not a population or success estimate.

`configs/topdial-semantic-strategy-debug-v9.json` is an additional opt-in candidate that excludes farewell guidance as well as ambiguous guidance. v8 remains unchanged for comparison. Both candidates retain the same ordered states, embeddings and MRN weights; the exclusion changes eligible map references, so each needs fresh graph/coverage calibration and both-GPU planner replay gates. v9 has not run. The additional discovery candidate adds 256 dialogues to the planned development matrix; recheck measured method-specific remaining-clock feasibility before launch rather than assuming the original reserve covers it. Select the candidate using discovery, then freeze it before disjoint confirmation.

The first slot-1 v8/v9 smokes under `daed880` passed functional controller/planner/journal checks, but inherited default GPU and zero H100-seconds fields in their run manifests. Preserve those original artifacts as functional smoke evidence with incomplete stage-compute provenance. The recorder now stores the actual allocated GPU and measured stage wall time on success/failure, retaining known failed-request counters. Fresh recording-v2 smoke namespaces on both GPUs are required before development evaluation; do not reuse old default compute as measured cost. Import and pre-stage audit overhead remain separate from the measured stage interval and must be accounted for using Slurm segment evidence.

Confirmation execution

Both candidates explicitly pin the previously authorized three concurrent Full
candidate branches. Fresh gates bind that execution count, code, config, package
lock, map/head and overlay provenance; a gate from another candidate is rejected.
The original smoke artifacts remain preserved. GDN batching can change generated
text; no serial/concurrent text equality is claimed.

After a reproduced discovery report shows positive point differences for both
methods, freeze `confirmation-candidate.lock`. Run the same committed code:

```bash
runtime/.venv/bin/python scripts/debug_topdial_planner.py --config configs/topdial-semantic-strategy-debug-v9.json --slot 0 --mode evaluate --variants-only repair-MRNFixed-R --confirmation-lock artifacts/CANDIDATE/confirmation-candidate.lock --output artifacts/CANDIDATE/confirmation-r
runtime/.venv/bin/python scripts/debug_topdial_planner.py --config configs/topdial-semantic-strategy-debug-v9.json --slot 1 --mode evaluate --variants-only repair-MRNFixed-Full --confirmation-lock artifacts/CANDIDATE/confirmation-candidate.lock --output artifacts/CANDIDATE/confirmation-full
runtime/.venv/bin/python scripts/evaluate_difftod_development.py --fitting artifacts/difftod-topdial-fit-v1-58159 --slot 0 --mode evaluate --gates artifacts/difftod-development-v1-58159/smoke-0 artifacts/difftod-development-v1-58159/smoke-1 --confirmation-lock artifacts/CANDIDATE/confirmation-candidate.lock --output artifacts/CANDIDATE/confirmation-difftod
runtime/.venv/bin/python scripts/compare_difftod_development.py --difftod artifacts/CANDIDATE/confirmation-difftod --mrn artifacts/CANDIDATE/confirmation-r artifacts/CANDIDATE/confirmation-full --confirmation-lock artifacts/CANDIDATE/confirmation-candidate.lock --output artifacts/CANDIDATE/confirmation-paired
```

Use the actual selected v8/v9 configuration, never switch after freezing. Planner
outputs share the candidate directory containing both fresh `smoke-0`/`smoke-1`
gates. Confirmation verifies the exact ranks-80:144 cohort and content hashes,
code/package/config/head/policy pins and both protocols. It uses locked
`H_edge`/`H_cover` directly: no training-pair or development coverage recalibration,
and no calibration embedding requests. The baseline gets a separate confirmation
request namespace. The report verifies all three sources and frozen thresholds.

The membership resource retains historical fixed-PPDPP metadata; the new candidate
lock separately pins the fitted DiffTOD policy. This is additional development
confirmation after earlier official testing, not fresh official-test evidence.
Single-seed scenario-paired CIs do not measure training-seed variability. A positive
macro paired-CI lower bound for both methods is the confirmation success flag;
absence of that evidence leaves superiority unproven. Official tests stay closed
for selecting repairs; human and full-paper gates remain separate obligations.

## Observed next-move candidate (v10)

`configs/topdial-semantic-strategy-debug-v10.json` retains v9's frozen head,
training map eligibility, exact-target/goal-type transfer, root seed, discovery
cohort, protocols and call budgets. It additionally supplies the retrieved
training assistant's observed next utterance alongside its semantic strategy.
The realization prompt restricts this example to conversational form and intent;
current knowledge and history remain the only permitted factual sources. No
reference history, future development turns or outcome labels enter the prompt.
This is a guidance adaptation candidate, not a measured improvement. Reference
facts may still contaminate realization despite the instruction, so inspect
outputs for unsupported copied names and claims before drawing conclusions.
Both physical GPUs require fresh runtime and exact planner replay smoke, then a
new measured feasibility projection before discovery evaluation. The untouched
confirmation cohort remains sealed until a candidate passes discovery and is
immutably frozen.

## Target-aligned recommendation candidate (v11)

v11 additionally excludes training references whose semantic action is
`recommend` but whose observed next assistant move does not mention that
training dialogue's declared target. The existing training `goal_reference`
field supplies this mention marker; it is never interpreted as acceptance or
successful completion. Asking, clarification, acknowledgement and explanation
references remain eligible, including intermediate states. Source states,
embeddings, terminal labels and fitted MRN weights remain immutable. This
heuristic can exclude legitimate implicit recommendations and does not prove
independent plan applicability. It requires fresh graph/coverage calibration,
both-GPU runtime/replay gates and measured remaining-clock feasibility. v10
results remain unchanged and confirmation outcomes remain sealed.

## Current-target factual grounding candidate (v12)

v12 returns to v10's map eligibility and observed guidance. It adds a grounding
section containing only triples from the current visible knowledge whose subject
matches the current target under the existing NFKC/case/whitespace normalization.
It does not add information unavailable to DiffTOD or change the shared assistant
policy, simulator, outcome judge, decoding, seeds, distance head or call budgets.
The guidance reminds realization to preserve subject bindings and acknowledge
missing target details. The original complete visible context remains available
for questions about other topics. This is an explicit planner-guidance adaptation,
not factual correctness enforcement: the language model may still invent or
misattribute facts, and output inspection remains necessary. Metadata parsing
rejects extra context fields and malformed triples; zero target triples is an
explicit empty fact list, never an external lookup or model fallback. Fresh
both-GPU runtime/replay and remaining-clock feasibility are required before
paired discovery. Confirmation remains sealed until a candidate is frozen.
