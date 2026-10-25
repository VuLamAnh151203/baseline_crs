# Guidelines implementation audit — 2026-10-12

Source: `TDRL_WWW_Feedback_and_Experiment_Guidelines.pdf` (10 pages), SHA-256 `af54dd69e5fe8431302e21bf99e28587cf137c8ae4f0d7bcda32c301834fad82`. Inspected source commit `1dd0169`.

Read-only implementation/evidence audit; no new inference or protocol changes. PDF recommendations do not override user-authorized scope.

This checklist uses 36 bounded requirement groups. Counts are audit categories, not a percentage of scientific completion. Fully implemented items do not imply causal validation or superiority.

| Status | Count |
|---|---:|
| Implemented | 8 |
| Partial | 11 |
| Not implemented / no experiment located | 16 |
| Explicitly excluded by user | 1 |

**Verification boundary:** Local report manifest and all seven report output hashes verified. Report aggregates account for 4000 dialogues across 16 cells. Full remote test records and head artifacts were not re-audited because SSH banner exchange timed out; local development reports are available but their remote source records are not mirrored locally.

## Requirement-by-requirement inventory

| ID | PDF page | Requirement | Status | Evidence and remaining limitation |
|---:|---|---|---|---|
| 1 | 2 | Correct positive-cost Bellman signs, goal/predecessor masks, detached targets and separate success reward | Implemented | src/tdrl/learning.py; tests/test_core.py; configs/comparison-judge-v2-sample500-seed17.json. Current c_step=1 and gamma=1 replace historical appendix parameters; no cosine-bound saturation claim is valid. |
| 2 | 1,5,8 | Directed geometry used for retrieval, graph edges, coverage and leaf scoring | Implemented | src/tdrl/reachability.py; graph.py; planner.py. Shared implementation is not evidence that sharing improves performance. |
| 3 | 3 | Both datasets, train/dev separation, immutable membership and duplicate audit | Implemented | data/processed/data-manifest.json; src/tdrl/official_data.py; near_duplicates.py; test_sampling.py. Audit establishes recorded exact/Jaccard checks, not absence of every semantic duplicate. |
| 4 | 3,4 | Shared response backbone, context, turn cap, simulator and stopping protocol | Implemented | configs/comparison-judge-v2-sample500-seed17.json; src/tdrl/evaluation.py; report manifest. Same-family simulated users are not live-user evidence. |
| 5 | 2,3 | Distinguish designated-target acceptance from mention or acceptance of another item | Implemented | src/tdrl/judge.py; JUDGE_SCHEMA and indexed observed evidence validation. Human validation of these judgments remains missing; graph anchors have different semantics. |
| 6 | 2,3,4 | Actual experimental seeds and scenario-paired uncertainty | Implemented | src/tdrl/metrics.py; artifacts/two-gpu-58159/report/paired-*.json; experimental_seeds=[17]. Scenario-paired CIs do not quantify training-seed variability. |
| 7 | 3 | IDs, model/prompt/config provenance, routing/candidate traces and replay | Implemented | src/tdrl/provenance.py; backend.py; stages.py; resources/test_scenarios_500_seed42.json; selection/selection.lock. Raw remote journals not re-downloaded in this audit. |
| 8 | 4 | Complete main-results comparison on both datasets | Partial | artifacts/two-gpu-58159/report/report.json: PPDPP/R/Full, both protocols, 500 scenarios, root seed 17. Scoped comparison exists; guidelines' full baseline set and quality columns do not. |
| 9 | 2,4 | Faithful closest baselines and adaptation disclosure | Partial | src/tdrl/ppdpp.py; policy_training.py; docs/BASELINES.md; PPDPP_REWARD_AUDIT.md. PPDPP implemented with disclosed weak CRS labels. DPDP/TRIP/DiffTOD only pinned/design; PDF ProCoT/EnPL/T-EPL/SCOPE not implemented and superseded by user baseline priorities. |
| 10 | 3,4 | SR, all-dialogue AvgT, measured calls/tokens/H100 seconds and paired SR differences | Implemented | src/tdrl/metrics.py; comparison.py; artifacts/two-gpu-58159/report/report.json. Cost currently describes evaluation-loop compute, not isolated serving cost. |
| 11 | 2,3,4 | Success-by-turn curves and successful-dialogue turns | Not implemented / no experiment located | metrics.aggregate and report export inspected. Actual turn counts are recorded but these requested summaries/curves are not generated. |
| 12 | 3,4,6 | Naturalness, suitability, preference consistency, coherence, factuality and pressure metrics | Not implemented / no experiment located | Judge labels and refusal flags exist; no quality scoring/report pipeline located. Refusal guard or success judge alone is not dialogue-quality evaluation. |
| 13 | 4,9 | Separate serving, simulator/evaluation-judge, offline training and graph-construction costs | Partial | Role-tagged request traces and stage compute exist; evaluation.py: actual_gen and elapsed. Report totals include outcome judge and simulator; no serving-only total or latency panel. |
| 14 | 2,7 | Explicit coverage/recovery denominators and measured re-entry/later success | Partial | planner.py logs covered/recovery; evaluation.py records actual_turns and full-budget turns. No aggregate reconciliation panel or recovery-success definition/report located. |
| 15 | 5 | R, random routing and always-rollout ablations with realized budget controls | Partial | PlannerConfig routing supports retrieval_only/random/always; R comparison exists. Random/always experiment results not located; one-call matched Full disables rollout. |
| 16 | 5 | Direct guidance without Dijkstra and uniform edges at fixed topology | Not implemented / no experiment located | graph.py/planner.py and executable experiment configurations inspected. No frozen ablation adapters or paired evidence. |
| 17 | 5,8 | Frozen generic, capacity-matched contrastive/temporal and Bellman representation controls | Not implemented / no experiment located | reachability.py and training.py inspected. MRN versus IQE does not isolate Bellman supervision from other training objectives. |
| 18 | 5 | Same-candidate alternative scorers and best-of-K without lookahead | Not implemented / no experiment located | planner.py leaf scoring and debug configs inspected. Nearest-reference guidance diagnostic is not the requested same-candidate scorer comparison. |
| 19 | 5 | Router/scorer interaction or separately trained detector for shared-geometry claim | Not implemented / no experiment located | No executable interaction/detector experiment located. Sharing one head across roles does not establish a benefit. |
| 20 | 6 | Human study preparation with institutional determination and pilot/power planning | Partial | docs/EXPERIMENT_PLAN.md human-study protocol. General safeguards and fixed 100/dataset design exist; determination ID, pilot, power analysis and operational recruitment packet not located. |
| 21 | 6 | Blinded human evaluation, ties, agreement and clustered uncertainty | Not implemented / no experiment located | No current human annotations/results located. Historical draft's 50-dialogue study is not evidence for this rebuild. |
| 22 | 6 | Stratified human audit of automatic successes/failures and moderator errors | Not implemented / no experiment located | No moderator-human agreement artifact located. No annotations may be collected before institutional determination. |
| 23 | 7 | Paired graph-versus-rollout interventions at identical prefixes with independent repeated continuations | Not implemented / no experiment located | Current evaluation runs separate full dialogues; no intervention experiment runner located. Paired scenario IDs do not create paired identical-prefix interventions. |
| 24 | 7 | Intervention-benefit routing controls: random, turn index, generic distance, confidence and oracle | Not implemented / no experiment located | Random routing primitive exists; no benefit-estimation/control experiment. No estimate of Delta(s), harmful-intervention rate or benefit at fixed intervention budget. |
| 25 | 7 | Relevant-support removal versus equal-size random/unrelated removal | Not implemented / no experiment located | No controlled support-removal manifests or runner located. Exact-target reference-pool filtering is not a controlled support-removal experiment. |
| 26 | 8 | Held-out reachability geometry and forward/reverse/horizon diagnostics | Partial | training.py:heldout_temporal_diagnostics; head numerical diagnostic code and dev reports. Chronology covers only part of required target mismatch, rejection, preference conflict and feasible cross-trajectory pair set; remote head evidence not re-audited. |
| 27 | 8 | Repeated-controller reachability and graph-edge execution validation plus blinded feasibility judgments | Not implemented / no experiment located | No edge-execution/reachability annotation experiment located. Quasimetric invariants do not prove an edge is executable in dialogue. |
| 28 | 8 | Collapse diagnostics: embedding variance, pair-distance distributions and hard confounders | Partial | Invariant/finiteness and chronological forward/reverse means exist. Complete variance/distribution/confounder panel not located. |
| 29 | 2,8 | Own-history failure prediction at fixed early turns 1/2/3, active counts and prevalence | Not implemented / no experiment located | No early-prefix prediction runner or report located. Completed-dialogue analyses and generic observed_prefix utility do not satisfy this requirement. |
| 30 | 8 | Prediction controls and AUROC/AUPRC, recall, lead time and leave-one-agent-out transfer | Not implemented / no experiment located | No classifier/control/evaluation pipeline located. Held-out remaining-step correlation is not early-failure AUROC/AUPRC. |
| 31 | 9 | SR-cost frontier, latency p50/p95 and independent H_edge/H_cover/K/L/bank sweeps | Partial | Selection threshold grid and runtime/request telemetry exist. No multi-budget frontier or K/L/bank sweep; development preflight p95 is not the full final-method latency report. |
| 32 | 9 | Explicit fallback for unseen targets and separate unfamiliar-combination evaluation | Partial | training.make_graph exact_target_when_supported; target-pool TopDial diagnostics. Goal-type transfer fallback is explicit in new debug path, but targeted unseen-target robustness comparisons on both datasets are missing. |
| 33 | 9 | Different-family lookahead/evaluation simulator transfer | Not implemented / no experiment located | One locked Qwen family throughout; profile/resistant protocols. Requires a separately authorized model/runtime amendment; current same-family simulation cannot establish this transfer. |
| 34 | 9 | Independent preference changes, constraints, topic detours and rejecting-user perturbations | Partial | evaluation.simulator_prompt supports frozen resistant profile. No independently specified perturbation grid or performance-change table. |
| 35 | 9 | Second response-backbone generator transfer | Explicitly excluded by user | User expressly excludes cross-LLM-backbone comparisons; AGENTS.md and EXPERIMENT_PLAN.md. Intentional scoped exclusion, not a bug; do not claim generator transfer. |
| 36 | 1,10 | Artifact-backed main/ablation/human/mechanism exhibits and coherent venue/paper delivery | Not implemented / no experiment located | Scoped JSON/Markdown report exists; tdrl/paper is empty. No new paper, required exhibit package, format validation or inspected PDF; guidelines assume WWW while repository method scope says EMNLP 2026. |

## Inconsistencies and interpretation risks

### I1 — scientific semantics

Graph goals are pre-response states whose next assistant utterance contains the target string, not independently judged user-acceptance states.

Evidence: `src/tdrl/training.py:47 and 377`.

Distance/path progress is progress toward a target-mention guidance anchor; it cannot by itself be interpreted as distance to successful acceptance.

### I2 — learning semantics

Censored cross-trajectory pairs are explicitly pushed to distance >= negative_margin despite no demonstrated unreachability.

Evidence: `training.PairSampler samples 25% cross-trajectory negatives; learning.ReachabilityLearner.step applies hinge separation`.

Prose says not universally unreachable, but optimization still repels potentially feasible cross-trajectory states. Need validated negatives, sensitivity or narrower interpretation; not proof every negative is wrong.

### I3 — compute comparison

Matched Full has generation_call_cap=1 while three branches x two lookahead require a planned 12 calls; recovery is disabled.

Evidence: `configs/comparison-judge-v2-sample500-seed17.json; planner.py:141-147`.

This is a disclosed one-call-cap fallback control, not a useful full-planner matched-compute comparison; calls cap does not match tokens/H100 seconds.

### I4 — cost interpretation

Dialogue generation totals and wall/H100 seconds include the outcome judge and evaluation user simulator.

Evidence: `evaluation.py:260,282,326-340; comparison.report`.

Valid evaluation-loop resource accounting, but cannot be labeled isolated serving cost without role/time separation. Guidelines exclude offline evaluation judges from serving cost.

### I5 — algorithm description

PDF describes minimum distance over lookahead; current Full scores final leaf only, with entry distance plus remaining directed path cost in revision 2.

Evidence: `PDF p5; planner.py:232-246`.

Document the intentional rebuild difference and compare scorer variants before attributing the historical mechanism to current code. Max negative value and min cost remain equivalent.

### I6 — terminal handling

Real dialogues use semantic outcome/refusal judgment, but hypothetical rollout branches stop via lexical refusal and never semantic goal achievement.

Evidence: `evaluation.py:260 and 317; planner.py:208`.

A hypothetical accepted branch may continue and be ranked by reachability instead of acceptance. This is a code-level risk; frequency/causal contribution is not yet measured.

### I7 — documentation

STATUS.md dated 2026-10-03 still says no H100 smoke, checkpoints, lock or report; EXPERIMENT_PLAN retains original deterministic-acceptance and one-GPU/three-seed/full-test plan.

Evidence: `docs/STATUS.md; docs/EXPERIMENT_PLAN.md; complete report manifest and later AGENTS amendments`.

Historical snapshots are not authoritative current status. Their assertions conflict with newer artifacts if read as current instructions.

### I8 — ablation isolation

Full does not guarantee retaining the exact R response as a candidate; candidate 0 and R use different derived seed purposes.

Evidence: `planner.py candidate generation versus /final seed path`.

Full-versus-R compares a recovery mechanism that also changes realized response candidates. It is not a claim that Full must always win; preserve seeds/budgets when redesigning controls.

## Authorized scope changes, not bugs

- 500 fixed scenarios and one experimental seed supersede PDF approximate 2000-scenario suggestion and old three-seed plan by explicit user request.
- User baseline priority is PPDPP/DPDP/TRIP/DiffTOD; PDF also recommends ProCoT/EnPL/T-EPL/SCOPE. Do not silently expand or relabel baseline scope.
- Cross-backbone comparison is excluded; different-family simulator study is outside current pinned local-Qwen runtime contract.
- Current debugging focuses MRN only; retain IQE historical evidence rather than claim current exhaustive head comparison.
- Guidelines' historical cosine and Bellman parameters are review context, not current runtime specifications.
- WWW positioning in the PDF and EMNLP 2026 in AGENTS require a venue decision before paper delivery; this audit does not change the venue.

## Next priorities

1. Finish MRN discovery diagnostic and validate leaf/goal semantics using the frozen local judge before another planner change.
2. Implement identical-prefix paired intervention with independent continuations; do not substitute full-dialogue scenario pairing.
3. Implement own-history early-prefix failure prediction and controlled support removal.
4. Generate missing turn/denominator/role-separated cost summaries from immutable traces; add random/always routing controls at meaningful measured budgets.
5. Prepare pilot/power and institutional human-study determination; no label collection until approved.

Machine-readable inventory and counts: `guidelines-audit-2026-10-12.json`. No historical paper numbers were reused. No inference, training, cancellation or protocol amendment was performed by this audit.
