# PPDPP / TDRL-R / TDRL Full automated comparison

Status (2026-10-05): allocation `57166` passed H100 Gate 2 and completed embedding banks,
directed-head training, both PPDPP SFT stages, and TopDial RL. DuRecDial RL stopped after
243/256 committed episodes; selection/test remain unstarted. See `DURECDIAL_DEBUG.md`.
The job is a three-method scope requested on 2026-10-03, not completion of the
repository's four-baseline, independent-coverage, transfer, human-study or paper obligations.
TDRL remains a temporary method name.

Canonical command, inside the one new Slurm allocation:

```bash
runtime/.venv/bin/python -m tdrl comparison \
  --config configs/comparison.json --output artifacts/allocation-$SLURM_JOB_ID
```

Local bootstrap/sync/submission command, after committing the source:

```bash
bash scripts/submit_comparison.sh
```

This script first checks SSH connectivity and local checks, copies only this new
checkout and official raw archives, installs the locked runtime, downloads/verifies
pinned weights, inspects the live scheduler, then submits a new allocation. It never
attaches to or cancels existing jobs. The allocation requests one full 80 GB H100,
160 GB host RAM and at most 20 days. Bootstrap executes no model inference.

Stages execute serially: repository/data checks; real dual-resident BF16 runtime
smoke; training-only Qwen embedding banks; separately trained MRNFixed/IQE per dataset;
PPDPP supervised training and on-policy RL per dataset; development threshold/head
selection; immutable scoped selection lock; official test; generated reports. A failed
stage stops the controller and retains partial evidence. Head and SFT checkpoints preserve
optimizer/RNG/cursor state; RL additionally saves pre-episode RNG and verifies reconstructed
sampled actions/gradients before resuming. Completed dialogue turns and requests are durable.
Unknown in-flight HTTP outcomes fail closed; they cannot be counted as resolved or silently
reissued. Gate 2 dialogue interruption/replay passed in allocation `57166`; recovery of the real
RL interruption still requires an audited amendment and H100 validation.

The primary test scope includes every official TopDial seen/unseen and DuRecDial 2.0
English test scenario, seeds 17/29/43, and identical profile/resistant-user protocols.
Test bodies remain sealed until selection. Train/dev archives are audited for exact and
word-trigram Jaccard >= .9 overlap before fitting. Future turn-aligned DuRecDial knowledge
is removed from initial context. Initial user observations and declared task targets are
shared; completed reference test conversations supply neither maps nor future dialogue.

PPDPP independently implements the native pooled RoBERTa policy, dropout .5,
categorical action classifier, supervised cross entropy and normalized-discounted-return
REINFORCE. Its intrinsic encoder is pinned separately in `policy-model.lock.json`;
all generation and simulation use the same locked Qwen. Recommendation adaptation uses
six disclosed weak strategy labels, a bounded visible target/profile header, deterministic
explicit-acceptance reward instead of the native graded critic, and dropout-disabled argmax
at evaluation. These departures do not establish equivalence to native PPDPP tasks.
The policy source and independent mathematical/control tests are linked in `ppdpp.py`.
Real SFT/RL checkpoints and fidelity manifests are required before scoped selection.

Natural full TDRL uses three candidates and two lookahead steps. The additional matched
configuration imposes one planner generation call per turn, with the same 128-token
per-call maximum across methods; PPDPP and TDRL-R natural traces already meet this ceiling
and are reused with explicit alias metadata. Full TDRL cannot fit candidate lookahead into
that ceiling, so it logs budget exhaustion and uses guidance realization. This is a budget
ceiling control, not evidence of equal realized tokens or equal H100 seconds. Reports must
show actual calls, tokens and allocated wall seconds; never describe the ceiling as strict
realized compute matching or that degraded variant as unconstrained Full TDRL.

Development selects 16 hash-fixed scenarios per dataset; four independently parameterized
edge/coverage threshold configurations are tested per head, using profile simulation.
Resistant test prompts and all decoding are frozen by the lock. Directionality/invariant
failures stop fitting; zero development successes leave the compute-per-success selection
rule unresolved and stop selection. The scope-specific selection schema v2 accepts PPDPP
fidelity only; schema v1 retains all four baseline requirements, and unknown versions fail.

After the first locked test probes, `eta.json` reports a projection from observed warm
p95 times with a 1.25 factor and a 24-hour reserve. This is an estimate, not a guarantee;
unobserved dataset/protocol conditions remain a limitation. If the full required scope
cannot fit the remaining allocation, the job stops with partial artifacts and requires an
explicit scope/budget decision. A 20-day wall-time request is a cap, never an ETA.

Reports are generated only from complete scenario/seed matrices, with exact official scenario
membership and a verified lock. They include individual-seed/aggregate SR, turns, compute,
and scenario-paired 95% bootstrap intervals with 10,000 resamples. No historical results,
partial shards or smoke measurements become scientific comparison claims. Cold server load
and physical GPU telemetry are saved separately; allocated/reserved counters report the
largest owned worker peak, with individual worker files, not a fabricated simultaneous sum.

Remaining evidence obligations: post-amendment RL recovery validation, development selection
and test evaluation; independent coverage labels; budget matching by all actual resource axes; bidirectional
transfer, complete quality/moderation reporting; full four-baseline/control matrix; approved
blind human study and paper. No human collection is authorized by this runner. These remain
pending even if the scoped automatic comparison completes.
