# Implementation and evidence status

Date: 2026-10-03. These are implementation statuses, not scientific results.

| Area | Completed local work | Scientific evidence / limitation |
|---|---|---|
| Repository | Package installed; core/runtime locks; model/source pins; static and CPU tests | H100 scientific stages remain unvalidated |
| Data | Official TopDial/DuRecDial English train/dev adapters; archive/input/membership hashes; exact/near audit | Test bodies sealed until selection; post-lock contamination audit implemented but not run |
| Directed learning | MRNFixed/IQE; positive-cost detached Bellman; failed/censored separation; sampler and checkpoints | No trained H100 checkpoint; real invariant/stability and held-out temporal artifacts pending |
| Maps/planning | Train-only directed maps; ordered retrieval and leaf scoring; conditional recovery; sticky refusal | No scientific map or interactive dialogue run |
| PPDPP | Independent pooled RoBERTa/dropout/classifier; SFT and on-policy REINFORCE; RNG/action gradient replay | Weak-label/reward/target/evaluation adaptations disclosed; no real policy checkpoint/fidelity evidence |
| Runtime gate | Full assertion runner: dual residency, live embeddings, covered/off-map lookahead, dialogue/resume/replay, GPU/allocator/call telemetry | No H100 smoke passed; precision/residency changes require user decision if the gate fails |
| Selection/test | Scoped v2 immutable lock; both datasets/three seeds/two protocols; natural and one-call-cap control; full membership checks | No lock or test results; strict actual token/H100 budget matching remains unresolved |
| Reports | Artifact-backed JSON/Markdown, per-seed aggregates and paired 10,000-resample CIs | No generated scientific report; quality/moderation/coverage/transfer evidence incomplete |
| Job | Sequential gated controller and one-H100, <=20-day batch; bootstrap/sync/submit script | Live job ID/state/ETA must be checked in the scheduler and submission artifacts |
| Overall project | PPDPP/DPDP/TRIP/DiffTOD remain the required project baseline list; cross-backbone comparisons excluded | DPDP/TRIP/DiffTOD, other controls, approved human study and paper gates pending |

The current user requested PPDPP versus TDRL-R versus TDRL Full. The scoped automatic
job does not declare completion of the overall required matrix or human/paper evidence.
TDRL is a temporary method name. Mandatory provider packages are permitted only as
transitive dependencies of the pinned local vLLM server; application code uses neither SDK.

New code/tests map the directed-consistency concern to `reachability.py`, `learning.py`,
`training.py` and held-out chronological diagnostics; simulation/autonomy to shared
`evaluation.py`, `acceptance.py`, `autonomy.py`; compute/baseline concerns to `ppdpp.py`,
`policy_training.py`, `stages.py` and `comparison.py`; reproducibility to versioned manifests,
request journals/ledgers, checkpoint replay and immutable scoped selection.
Independent support labels, transfer and approved human judgments remain separate evidence
obligations. Every real stage must retain artifact paths/run IDs and its unresolved limitations.
No historical result has been imported or reused. CPU tests establish control flow only.
