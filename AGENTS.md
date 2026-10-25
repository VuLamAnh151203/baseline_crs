# AGENTS.md

## Method and scope

This repository rebuilds the EMNLP 2026 target-driven conversational recommendation project from first principles. The proposed planner learns a **directed quasimetric** over frozen dialogue-state embeddings to estimate the cost of reaching a reference state. The same distance weights directed graph edges, retrieves guidance, determines retrieval-defined plan coverage, and scores rollout leaves. Covered states follow offline shortest-path guidance; unsupported states trigger candidate generation and lookahead to return to a guidance-backed region. Training resources build the offline map, development selects the configuration, and locked official test splits provide evaluation.

**TDRL (Temporal Distance Reinforcement Learning) is a temporary placeholder, not the final method name.** Use it consistently in code and drafts until the user selects a final name; do not imply naming is finalized. Implement the local project under `./tdrl/`.

These instructions apply repository-wide. Historical repositories `../crs/TDRL_OR` and `../crs/emnlp26_crs` are read-only context for interfaces, failure modes, and reviewer concerns, never valid experimental evidence.

## Reviewer concerns and rebuttal fixes

Source: the local `Stop Blindly Following the Map_ Coverage-Aware Conversational Planning via Learned Reachability _ OpenReview.pdf`. These are **required rebuild fixes and evidence obligations**, not completed-work claims. Historical rebuttal numbers and promises do not satisfy them.

- **Directionality/Bellman consistency (LgSA, f5sU):** make MRNFixed/IQE directed quasimetrics primary and preserve argument order throughout planning. Learn positive step-cost distance, separate success reward from reachability cost, and test goal/predecessor/failed-terminal targets. A directed graph alone does not resolve symmetric-distance objections.
- **Simulation validity (LgSA, f5sU):** qualify results as interactive simulation evidence; run a frozen resistant-user protocol across methods. Versioned local Qwen semantic outcome judgment is primary under the user-authorized 2026-10-07 judge amendment; retain historical deterministic labels as historical protocol only. Human dialogue-quality judgments do not establish live-user deployment performance.
- **Quality/autonomy (f5sU):** measure naturalness, coherence, appropriateness, and pushiness alongside success; stop pressure after refusal. Expand blind human evaluation to both datasets with institutional determination and transparent recruitment, payment, consent, and data handling.
- **Compute/dynamic baselines (LgSA, f5sU, AC):** compare PPDPP, DPDP, TRIP, and DiffTOD with the same locked Qwen backbone at natural and matched budgets; include random routing and always-rollout controls. Report actual calls, tokens, latency, VRAM, and H100-seconds per dialogue/success. Exclude cross-LLM-backbone comparisons. Preserve each baseline's intrinsic policy/planning mechanism and document dataset adaptations; do not relabel prompt-only approximations as faithful baselines.
- **Non-circular coverage/transfer (f5sU):** use independent applicability labels, controlled support holdouts, and temporal/remaining-step diagnostics. Separate and development-calibrate `H_edge`/`H_cover`; report sensitivity and bidirectional dataset transfer. Claim unsupported retrieval states and pre-intervention failure risk, not universal OOD or intervention-invariant prediction.
- **Mechanism/reproducibility (LK2V, f5sU):** isolate representation, routing, and recovery; select covered/off-map/recovery-success/recovery-failure examples by a documented rule. Release reproducible code, prompts, manifests, and permitted artifacts with immutable model provenance; fix references and formatting.

For each rebuttal point, link concern → implementation/test → new artifact path/run IDs → remaining limitation. Distinguish proposed, partial, failed, and completed work; never reuse historical numbers.

## Non-negotiable runtime and evidence rules

- No hosted inference, inference credentials/cloud credential lookup, application use of hosted-provider SDKs, proprietary judges, or silent fallback. User-authorized exception (2026-10-03): mandatory transitive `openai`/`anthropic` packages may be installed only as dependencies of the pinned local vLLM server. Application inference code must never import these SDKs; use the loopback-only `httpx` client. This exception permits no hosted endpoint, credential lookup, or hosted inference. Local transport (e.g. `httpx`) must reject non-loopback hosts at runtime.
- Generation: **`Qwen/Qwen3.5-9B`**; embeddings: **`Qwen/Qwen3-Embedding-8B`**, normalized full **4096 dimensions**. Pin immutable revisions/weight hashes before scientific runs. Primary precision is **BF16**; no quantization or model substitution.
- Text-only, non-thinking generation: `chat_template_kwargs.enable_thinking=false`. Store final response text only; never put hidden reasoning into dialogue history.
- All generation/embedding inference runs on **H100 inside Slurm**. Serve via vLLM in the experiment allocation, bound to `127.0.0.1` or a Unix socket. Interactive evaluation requires both models resident on **one 80 GB H100** and live embeddings. Offline embedding jobs may load only the embedding model.
- If dual residency fails, retain measurements and stop at the runtime gate. Precision, quantization, model size, GPU count, or CPU placement changes require an explicit user decision.
- Downloads/installations are bootstrap, not inference experiments. Populate local caches so final jobs can run with network inference disabled.
- Never import legacy embeddings, checkpoints, graphs, traces, metrics, tables, plots, or numeric claims. No test tuning or fabricated progress; smoke tests, partial shards, and submitted jobs are not scientific results.

Official model cards: [generation](https://huggingface.co/Qwen/Qwen3.5-9B), [embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B).

## Repository, provenance, and interfaces

Use `configs/`, `data/` (manifests/preprocessing metadata), `docs/` (protocols/ethics/design), `paper/`, `scripts/` including `slurm/`, `src/tdrl/`, `tests/`, and normally gitignored `artifacts/`, or an equally clear existing layout. Each stage has one documented command and a resumable manifest-backed output directory. The implementation root is `./tdrl/`; scientific stages must run sequentially inside one new SSH H100 Slurm allocation, requesting one full 80 GB H100 and a wall-time limit no greater than 20 days. Bootstrap downloads/installations happen before allocation where permitted. Measure feasibility after Gate 2; if the full mandatory matrix cannot fit, retain partial artifacts and request an explicit scope/budget decision rather than silently dropping methods, seeds, datasets, protocols, or gates. Human determination/annotation is an external prerequisite, not GPU inference. User-authorized recovery exception (2026-10-05): recover allocation `57166` in one new H100 allocation, preserving the 243 committed DuRecDial PPDPP RL updates and completing the remaining 13 before continuing the scoped comparison. Retain original artifacts and the unknown failed embedding outcome; audit imported hashes, the transport/lifecycle and recovery-instrumentation code amendment, and exact action/gradient replay on H100 before another update. Report both allocation IDs and segment-specific compute; do not claim this recovered pipeline ran entirely in one allocation. User-authorized debugging/fix continuation (2026-10-06): retain failed recovery allocation `57561`, verify it made zero new updates and did not reissue the pending request, and submit a replacement recovery allocation. Diagnose default-backward versus strict deterministic-backward replay; record the controller-only numerical execution amendment and require exact repeated actions/log-probabilities/gradients and unchanged forward behavior before any update. User allocation instruction (2026-10-06): replace pending job `57710` with one persistent allocation requesting exactly one full H100 80 GB, 16 Slurm CPUs, 128G host RAM, and 20 days. Run the recovery pipeline and all subsequent jobs as sequential steps inside this allocation; keep the allocation after pipeline completion/failure for further authorized work, until released or its time limit. Record the active job ID and preserve scientific gates. User-authorized selection fix (2026-10-06): after the pre-lock request-ID collision, invalidate all selection artifacts/cache from `artifacts/recovery-57711/selection`, retain hash-verified completed fitting, and rerun fresh selection/test/report in the existing allocation `57711`. Audit the request namespace/control-flow amendment; preserve seeds, prompts, models, budgets and config, and retain the original allocation start time for feasibility. Fresh H100 smoke must verify distinct trial namespaces and exact replay before selection. User-authorized low-success repair (2026-10-07): debug both PPDPP and TDRL and reduce MRN/IQE engineering. Retain head architecture/checkpoint formulas and acceptance v1; audit cached latent geometry and offline terminal-mask compatibility on H100. Amend strategy-only guidance, shared assistant/simulator prompts, refusal v2 and zero-success/schema-v3 selection handling. In allocation `57711`, retain compatible offline embeddings/heads/SFT, rerun both PPDPP RL stages with the frozen original budgets/seeds, run paired PPDPP development diagnostics, and rerun fresh selection/test/report. Preserve all old artifacts and the allocation clock; do not discard non-recommendation dataset targets or claim improved SR before new evidence. This exception does not waive model, precision, scientific-contract, selection, test or human gates.

User-authorized PPDPP fidelity/judge repair (2026-10-07): stop the low-success pipeline at a durable checkpoint and retain all old artifacts. Restart under configs/comparison-judge-v2.json in allocation 57711 with its original clock; do not reuse old fitting or selection. Restore pinned-code sampled per-turn graded critic (10 samples, temperature 1.1), random training-case sampling, explicit legacy HF AdamW math, SFT warmup/linear schedule/no-decay groups and development macro-precision checkpoint selection, 1000 RL episodes/lr 1e-6/gamma .999, no RL gradient clipping and explicit action instructions. Use one deterministic local Qwen semantic outcome judgment per real turn across methods; JSON validation is transport validation, never utterance-regex success or fallback. BF16 policy forward uses FP32 optimizer masters/moments, audited exact action/logprob/gradient replay and dual residency. Offline trajectory ends without independent outcome evidence are censored, never regex-labelled successes; refit heads. Retain and disclose weak CRS strategy labels, target conditioning, shared role/turn/token/refusal protocol and the paper-versus-code reward distinction. Fresh H100 judge/controller smoke and measured budget preflight must pass before full fitting; stop and request an explicit scope/budget decision if projection does not fit. No change waives selection/test, model/precision, human or paper gates.

User-authorized test-scope amendment (2026-10-08): evaluate a pooled uniform random sample of 2000 total official test scenarios across both datasets, without replacement, fixed sampling seed 42. Freeze the sampling algorithm/config before selection; open test resources and persist immutable selected IDs, population/sample hashes and counts only after selection.lock. Reuse the identical subset across all methods, evaluation seeds 17/29/43 and both protocols; retain natural/matched budgets, training, development and other gates. Label reports as sampled official-test evaluation, never full-test completion. Run in allocation 57711 with its original clock and a fresh measured feasibility gate; preserve previous failed preflights.

User-authorized single-seed amendment (2026-10-08, supersedes the three-seed requirement for the current scoped job): run all fitting, development and test experiments with one root seed 17; retain sampling seed 42 exclusively for the fixed 2000-scenario test subset. Preserve both datasets, methods, natural/matched budgets, profile/resistant protocols and all gates. Derived per-request/candidate RNG streams and paired bootstrap resampling remain reproducible auxiliary randomness, not extra experimental runs. Retain the superseded three-seed branch, stop it at a safe boundary, and restart with the amended committed config in persistent allocation 57711. Report single-seed evidence explicitly; scenario-paired CIs do not quantify training or seed variability.

User-authorized concurrency amendment (2026-10-08): run PPDPP and natural-budget TDRL Full concurrently as two dialogue workers sharing the same local Qwen servers in allocation 57711; TDRL-R and matched-budget TDRL remain sequential. Preserve the 2000-scenario/seed-42 sample, experimental seed 17, all models/precision/prompts/decoding/budgets and scientific gate order. Stop the prior pipeline at a durable checkpoint, audit unchanged fitting modules and policy interfaces, retain hash-verified completed fitting and exact partial SFT optimizer/RNG/provenance, and record source/current segments. Before another update, require fresh dual-residency/controller/concurrency smoke, isolated request counters and exact completed-pair journal replay. Record GDN batch-dependent generation as an execution amendment, never claim serial/concurrent text equality. Allocate shared dialogue busy time by documented equal-active-worker convention without double counting; report cold/stage overhead separately. Measure development throughput/ETA and recheck post-fit method-specific feasibility; official test remains locked.

User-authorized 500-scenario amendment (2026-10-08, supersedes the 2000-scenario scope): sample 500 total official test scenarios across both datasets uniformly without replacement using fixed seed 42, shared identically by every method/budget/protocol. Keep experimental seed 17, both protocols, all training/development budgets and PPDPP/Full concurrency. Freeze the config before selection, persist IDs/hashes only after selection.lock, and label sampled-test evidence. At the durable checkpoint, preserve compatible completed fitting and hash-audited partial PPDPP RL optimizer/RNG/provenance; require exact H100 action/logprob/gradient replay before the next update. Retain original artifacts, segment compute and allocation 57711 clock; run fresh feasibility gates.

User-authorized two-GPU replacement (2026-10-08): cancel allocation 57711 after preserving its durable 120-update TopDial RL checkpoint and all completed fitting. Replace it with persistent allocation 58159 requesting exactly two full H100 80GB GPUs, 16 total CPUs, 128G host RAM and 20 days. Use two independent Slurm steps (one visible GPU/eight CPUs each), with both pinned BF16 Qwen models resident per GPU during interactive inference. Use disjoint loopback ports and UUID-verified GPU assignment. Schedule independent offline/policy/development tasks concurrently subject to gate/data dependencies; aggregate both datasets for one immutable global selection lock. Evaluate PPDPP/retrieval/matched and natural Full on separate GPUs with the identical 500-scenario sample (seed 42), experimental seed 17 and both protocols. Fresh runtime/controller/concurrency gates on both GPUs precede updates; preserve partial optimizer/RNG/provenance and exact replay. Record source/new allocation IDs, source compute and new per-GPU/cumulative H100-seconds; do not claim all fitting ran in the new allocation. Keep the allocation on pipeline completion/failure. The new hardware allocation clock supersedes the retired allocation clock for remaining feasibility.

Truth precedence: new machine-readable artifacts → frozen configs/manifests → code/tests → official dataset/model documentation → historical context. Every paper number resolves to an artifact; generated Markdown/LaTeX tables/figures record artifact paths and run IDs in source metadata. Never hand-edit generated numbers.

Use a versioned, validated run-manifest schema with explicit partial/complete status; unknown versions fail closed. Record:

- git commit/dirty state; config path/SHA-256; package lock hash; Python/CUDA/driver/vLLM/PyTorch versions;
- dataset/split/revision/input hashes; fixed scenario IDs/order; random, decoding, and candidate seeds;
- model IDs/revisions/weight hashes; prompt/template/preprocessing hashes; checkpoint hashes/architecture signatures;
- Slurm job IDs/logs; GPU model/count/UUID when available; peak allocated/reserved VRAM; start/end timestamps; cold/warm timing;
- generation/embedding request counts; input/output/total tokens by role; turn/dialogue latency; H100-seconds; errors/retries; downstream output hashes.

Reuse caches only within this repository with an exact full-provenance match; misses/mismatches fail closed. Never mix embedding revisions or preprocessing spaces.

Keep typed semantic interfaces:

| Interface | Required contract |
| --- | --- |
| `GenerationBackend.generate(...)` | Input: messages, role, decoding config, seed, request ID. Output: final text, token counts, seed/resolved decoding, latency/retries, model ID/revision, prompt/template hash. |
| `EmbeddingBackend.embed(...)` | Input: texts, task/instruction, batch ID, request ID. Output: normalized 4096-D vectors in stable source order, model ID/revision, instruction/preprocessing hashes, batch size, latency, peak-memory metadata. |
| `DirectedReachability.distance(source, target)` | Ordered cost `d(source,target)`; expose head type, checkpoint hash, encoder provenance, batch scoring; support MRNFixed and IQE. Never hide order in an unordered pair. |
| `Planner.act(...)` | Final response, covered/off-map decision, guidance evidence/directed distances, selected path/candidate, rollout/lookahead scores, generation/embedding/planner compute trace, replay IDs. |

## Reachability, data, and selection

Serialize states identically across training, graphs, retrieval, and evaluation using only information legitimately available at that turn. Prevent target/future-turn/success-label/test-resource leakage. Use frozen Qwen embeddings unless a declared ablation changes this; caches hash serialization, instruction, model revision, and preprocessing.

Implement **MRNFixed and IQE-compatible heads** with non-negativity, zero self-distance, asymmetry, and triangle inequality by construction. An unconstrained MLP is not a quasimetric. Cosine is only a symmetric control, generic retrieval baseline, or explicitly auxiliary clustering signal. References: [MRN](https://ojs.aaai.org/index.php/AAAI/article/view/26058), [IQE](https://www.tongzhouwang.info/interval_quasimetric_embedding/).

For valid transitions toward sampled reachable goals, learn distance directly using a documented detached target network:

```text
y(s_t, g) = 0                                           if s_t is g
y(s_t, g) = c_step + gamma * m(t+1, g) * d_target(s_t+1,g) otherwise
```

`c_step > 0`; `m` is zero when the next state attains the goal. A one-step predecessor has target `c_step`. Keep success reward and auxiliary step cost in separate types/config fields. Failed terminals and censored/unreachable pairs use a declared negative/separation objective, never a reachable terminal mask that falsely makes them one step away.

Use the **same selected directed model** for edges `d(u,v)`, retrieval `d(live,reference)`, coverage dispatch, and rollout leaf-to-guidance scoring. Never infer reverse edges. Run directed shortest paths; reject negative/NaN/infinite weights; preserve orientation in logs and figures.

`H_edge` controls connectivity; `H_cover` controls dispatch. Calibrate separately per dataset on development and freeze both. Report development curves, clearly post-hoc test sensitivity, and zero-shot TopDial↔DuRecDial threshold transfer. Never update thresholds from test outcomes.

Use official **TopDial and DuRecDial2.0** splits/licenses. Only permitted training resources build representations, banks, nodes, edges, and guidance; development selects/calibrates. Development scenarios query training-built maps, never supply graph trajectories. If no official dev split exists, deterministically carve one from training before fitting and persist IDs. Preserve raw immutable inputs, document filtering/translation/normalization/truncation, hash normalized examples, and audit exact/near duplicates. Test dialogues/outcomes/traces/human labels stay out of training/selection; open test resources only after selection is locked.

Compare heads with identical data, embeddings, optimizer/graph/rollout budgets, and development scenarios. Select once:

1. Discard heads failing invariants, stability, or provenance.
2. Prefer higher macro-average development SR across both datasets.
3. If the scenario-paired 95% bootstrap CI for the SR difference contains zero, prefer lower H100-seconds per success.
4. If compute differs by less than 1%, prefer fewer parameters; remaining tie → MRNFixed.

Before test, write immutable `selection.lock` with decision, all candidate metrics, thresholds, prompt hashes, budgets, and baseline adaptations. Test commands require and verify it.

Coverage means **retrieval-defined plan support**, not general OOD. Require labels made without learned-distance access, controlled trajectory/domain holdouts, annotation-based next-move/strategy compatibility where available, and held-out temporal-order/remaining-step diagnostics. Compare lexical, generic semantic, cosine, MRNFixed, and IQE on identical states/labels; report AUROC/AUPRC and positive prevalence. Do not validate only on the method's own split, cherry-pick test scores, or claim intervention-invariant failure prediction. Document qualitative case selection.

## Evaluation and human study

User-requested job scope (2026-10-03): the current automated comparison is PPDPP vs TDRL-R vs TDRL (Full), on both datasets with three seeds. This explicit scoped job does not claim completion of the full baseline/control matrix, human study, or paper gates below.

Required methods: Qwen without planning, TDRL-R (graph/retrieval only), full conditional-recovery TDRL, symmetric cosine control, budget-matched random rollout routing, always-rollout, PPDPP, DPDP, TRIP, and DiffTOD. Exclude cross-LLM-backbone comparisons. Document baseline adaptations; intrinsic learned policies are allowed, but all language realization/LLM planning use locked Qwen.

Freeze role-specific decoding on development. Methods share assistant realization/final-response length, simulator/moderator, success rules, prompt-visible information, scenario order, turn limit, and paired seeds. Record per-candidate rollout seeds. Evaluate adaptive planners at natural and matched budgets using **actual calls, generated tokens, and H100-seconds**, not nominal nodes; always-rollout is an upper-compute control.

- Full official test splits, **three controlled seeds**, per-scenario records, individual-seed and aggregate reports. Unsuccessful dialogues count at full turn budget. Report infrastructure errors separately; rerun paired methods on the identical valid scenario set.
- Primary success requires explicit, unpressured target acceptance; distinguish mention/recommendation/rejection/acceptance. Use the frozen local Qwen semantic outcome judge consistently across methods; require quoted user evidence, retain ambiguity and errors, and do not equate mention with acceptance. Task-specific non-recommendation completion is reported separately from recommendation acceptance. Historical deterministic labels are not overwritten.
- Metrics: SR, average turns over all dialogues, success-weighted compute/H100-seconds per success, rollout trigger/recovery success, frozen coverage-conditioned results. Use scenario-paired **95% bootstrap CIs, 10,000 resamples**, fixed seed, preserving pairing. Deterministic reruns are not independent human-like samples.
- Run identical profile-conditioned and frozen resistant-user protocols permitting skepticism, refusal, correction, and topic shifts. Qualify results as simulation evidence, not real-user deployment.
- Track naturalness, helpfulness, coherence, appropriateness, repetition after rejection, autonomy, refusal violations, and repeated persuasion. Stop pressure after explicit refusal under the frozen policy; success alone is not quality.
- Compute per method/dataset: generation/embedding calls per turn/dialogue; input/output/total tokens; embedding texts/tokens when available; warm p50/p95 turn latency; dialogue wall time; separate cold load time; H100-seconds/dialogue and success; peak VRAM; rollout trigger/candidates/lookahead; failures/retries. No fictional API cost; estimates must be separate with dated assumptions.

Human evaluation is mandatory on **both datasets**: 100 fixed test scenarios each (record seed); blinded TDRL vs TDRL-R and TDRL vs strongest non-TDRL baseline selected on development; independent A/B randomization manifest; three independent raters per pair. Hide names, metrics, traces, and outcomes not visible in dialogue. Judge naturalness, coherence, appropriateness, pushiness/autonomy, and overall quality; allow ties. Report win/tie/loss, paired uncertainty, agreement, and anonymized item-level judgments.

**Do not collect human labels before required institutional determination.** Document recruitment, inclusion criteria, annotator count/locale, instructions, quality controls, compensation/hourly rate, consent, data handling, and IRB/exemption status. Qwen judging is a separate diagnostic, never a substitute for or blend with human results.

## Gates and verification

Complete gates in order; never submit full experiments before the relevant smoke passes:

| Gate | Required evidence |
| --- | --- |
| 0 — Repository | Package/configs/tests, artifact schema, dependency lock/model pins, static checks, canonical stage commands. |
| 1 — Data | Split/license audit, manifests/hashes, duplicate/leakage audit, frozen serialization/membership. |
| 2 — H100 smoke | Both BF16 models on one 80 GB H100; launch commands/revisions/vLLM memory settings; idle/peak/steady VRAM; real 4096-D embedding/non-thinking generation; assistant/simulator turn, complete live-embedding dialogue, covered retrieval and off-map candidate/lookahead; interruption/resume without duplication and exact replay; manifest/no external inference evidence. Stop before full training if failed. |
| 3 — Learning | Identical-budget MRNFixed/IQE training; invariant/Bellman/finiteness/provenance tests; independent diagnostics; retain every candidate checkpoint/selection artifact. |
| 4 — Selection | Training-built banks/graphs evaluated on development; thresholds/prompts/head/budgets/adaptations frozen by selection rule; immutable lock. |
| 5 — Test | All baselines/ablations, three seeds/both datasets; compute/transfer/coverage/robustness reports. Record post-lock amendments; invalidate/rerun all affected paired methods after bugs. |
| 6 — Human/paper | Approved blind study completed; tables/figures regenerated; evidence-supported claims; built/inspected PDF. |

Required tests:

- Static checks on executable/configured inference paths: hosted endpoints/dependencies, credentials, legacy imports/results, and non-loopback adapter acceptance fail CI. Documentation may discuss forbidden providers.
- Unit/regression: serialization/no leakage; embedding shape/normalization/order/provenance; quasimetric invariants/asymmetry; Bellman goal/one-step/multi-step/failed-terminal targets, detachment, scale/finite gradients; directed edges/no reverse inference; live-to-reference order; separate thresholds; split/lock enforcement; seeds/replay; manifest versions/partial status; compute counters.
- Document BF16 tolerances tight enough to reject symmetric/unconstrained impostors. CPU/fake integration tests establish control flow only, never H100 evidence; real smoke covers Gate 2.
- Completion checks: expected scenarios/no duplicates, required fields, zero unresolved infrastructure errors, matching paired model/prompt/config contracts, expected seeds/lock, finite metrics/counters, checksums, regenerated table/figure inputs.

## Paper and working discipline

Write paper-facing content in English using the temporary name consistently. Explain directed argument order, retrieval-defined coverage, development-frozen thresholds, cosine's auxiliary/control role, full local-model/H100 budget, same-family simulation limits, and human-study scope/safeguards. Bound claims to this protocol and evaluated baselines; no universal OOD, guaranteed real-user performance, or unqualified superiority claims. Every abstract/introduction/conclusion/table/figure number must trace to new evidence.

Build with `latexmk`; inspect logs, PDF metadata/extracted text, references, table consistency, and rendered pages. Completion requires all core artifacts here, regenerated results, passed scientific/runtime gates, fair baselines, approved human evidence, and a clean paper build.

Read this file before work; inspect existing code/manifests before asking repository-answerable questions. Keep changes narrow, preserve unrelated work, update tests for behavior changes, and prefer resumable stages. Inspect live scheduler/process/GPU state before launch; never kill/reuse/attach to unknown allocations or others' processes. Store job IDs/logs; distinguish submission from completion. Report completed artifacts, partial work, failures, and proposals separately. If a non-negotiable constraint fails, stop at that gate and ask for an explicit decision; never invent a fallback.

User-authorized critic execution amendment (2026-10-09): run the existing ten
PPDPP training-critic samples concurrently on each worker's local Qwen server.
Preserve sample count, prompts, decoding, derived sample seeds, request IDs,
validation and index-ordered reward averaging. Retain durable fitting/RNG/optimizer
and completed development artifacts; disclose batch-dependent generation and the
source/current execution segments, without claiming serial/concurrent text equality.
Require dual-residency/controller and exact journal-replay critic smoke on both
GPUs in allocation 58159 before another update. Preserve allocation clock,
selection/test gates and the fixed 500-scenario/seed-42, experimental-seed-17 scope.

User-authorized candidate execution amendment (2026-10-09): run the three Full
candidate branches concurrently, preserving each branch's sequential lookahead,
seed/request IDs, prompts, refusal rules, index-ordered tie-break and budgets.
Retain compatible fitting/PPDPP checkpoints and source compute; preserve old
selection artifacts and rerun both datasets/heads in fresh selection namespaces
under the same concurrent execution before global locking. Allow the current
selection stage to finish while RL continues, then pause at a durable RL boundary.
Require real dual-residency/controller, MRNFixed/IQE branch and exact journal/plan
replay smoke on both GPUs before enabling the new execution in allocation 58159.
Disclose batch-dependent generation and source/current segments; preserve the
allocation clock, scientific config, 500-scenario seed-42 sample and root seed 17.

User-authorized fixed test membership amendment (2026-10-10): freeze the exact
500 official-test scenarios already used by completed allocation 58159, retaining
330 TopDial and 170 DuRecDial2.0 cases. Package explicit stable IDs, canonical
zero-based per-dataset indices, order, population/subset and normalized selected
scenario content hashes; replay that resource on all machines without resampling.
Reject missing/changed populations, content, IDs, indices or resource checksums.
Retain the original seed-42 sampling provenance, root seed 17, all paired method/
budget/protocol scope and immutable completed results. Test bodies remain gated by
a verified selection lock; fitting/development never use the pinned test resource.

User-authorized TopDial planner debugging (2026-10-10): investigate TDRL-R/Full
against fixed PPDPP using training/development, with no guarantee of superiority
and no selection from opened test outcomes. Preserve original fitting, locks,
reports and request evidence. Audit goal-type versus declared-target reference
conditioning, next-action guidance and leaf-to-goal directed path cost; retain
MRNFixed/IQE formulas and shared model/precision/simulator/judge contracts. Run a
frozen expanded 64-scenario development diagnostic (root seed 17, both protocols)
on allocation 58159 after new runtime/controller/planner replay gates on both GPUs.
This diagnostic does not replace the original 16-case selection/test budget or
claim independent post-test confirmation. Keep all goals, disclose goal-type
transfer when an exact training anchor is absent, and retain single-seed limits.

TopDial numerical debugging continuation (2026-10-10, within the authorized
planner-debug scope): retain the ongoing planner-v2 diagnostic unchanged. Audit
BF16 EMA update loss and BF16 optimizer moments against real head checkpoints;
CPU tensor arithmetic is numerical evidence only. A fresh numerical-revision-2
head experiment preserves BF16 model/input/forward parameters and MRNFixed/IQE
formulas, with FP32 auxiliary optimizer and target-EMA masters. Preserve all
training hyperparameters, data, root seed 17 and identical 30,000-step budgets;
require H100 exact forward/gradient/update replay before fresh fitting, original
invariant tolerances and new runtime/selection gates before new dialogue claims.
Keep original checkpoints and record segment-specific compute; never substitute
numerical diagnostics or a merely higher development point estimate for verified
TDRL-R/Full improvement over PPDPP on TopDial.

TopDial diagnostic namespace repair (2026-10-10): retain the 384 completed
baseline and 256 completed MRN development dialogues after output-hash audit.
The shared MRN/IQE diagnostic journal collision was rejected before HTTP dispatch;
require no unresolved pending requests, fresh variant/head/full-contract request
namespaces and fresh runtime/planner replay gates before the remaining 256 IQE
dialogues. Preserve collided journals and old manifests; use a new output segment.
Root seed, prompts, budgets, models and scenario order remain frozen. Independent
numerical refits may use the other authorized H100 after their own exact gate;
record both source and new compute without claiming a completed comparison early.

User MRN-only debugging scope (2026-10-12): from now on focus experimental
debugging on MRNFixed, comparing TDRL-R and Full against fixed PPDPP. Stop the
ongoing IQE evaluation step while preserving complete/partial dialogues, journals,
checkpoints and segment compute; continue the MRN worker without changing its
contract. Retain all completed IQE evidence and original global selection locks.
This narrows the current debugging continuation, not the historical results or
mandatory paper baseline matrix. Future diagnostics use MRN only, root seed 17,
both user protocols and unchanged primary outcome/simulator/model contracts.
