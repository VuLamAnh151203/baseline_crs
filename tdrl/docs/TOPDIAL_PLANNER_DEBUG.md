# TopDial planner repair diagnostic, 2026-10-10

User request: debug TDRL-R and TDRL to improve against PPDPP on TopDial.
Scientific aim is improvement, not a guaranteed or manufactured win. Official
500-scenario results have already been opened. This diagnostic reads only training
and development; it does not select settings from official-test outcomes. Keep
all old fitting, locks, dialogue records, reports and compute evidence immutable.

## Findings and amendments

1. The old graph goals were pooled by goal type; an actual movie/song could route
to an anchor for another item. Revision 2 reserves training goal-reference nodes
for the currently declared target when such nodes exist. If none exist, explicitly
log `goal_type_transfer`; do not discard unseen targets. No test trajectory or
future user acceptance enters the bank. A goal reference remains a training
mention/reference-state anchor, not an acceptance label or reward.
2. Old guidance concatenated up to three future path moves and omitted the explicit
six strategy instructions used by PPDPP. Revision 2 realizes the next reference's
observed strategy using the same frozen instruction dictionary. PPDPP's action
instructions, policy input/forward and exact realization text remain unchanged.
3. Old rollout leaves used only live-to-reference entry distance. Revision 2 scores
entry distance plus the remaining directed path cost, using the same fitted head
and recorded forward edges. Covered routing still tests nearest entry distance
against H_cover. Reference selection minimizes goal cost among supported nodes,
or among the available guidance references if unsupported. Recovery explores a
training-guided action, explanation and recommendation, with the unchanged three
candidates/two-lookahead budget and concurrent branch scheduling.

MRNFixed/IQE formulas, fitted checkpoints, embedding serialization/model revision,
PPDPP fitting and shared assistant/simulator/outcome-judge protocol are unchanged.
The bank's weak strategy labels and mention-based reference anchors remain material
limitations. An exact item name absent from training anchors is not inferred from
a future/test outcome; goal-type strategy transfer remains separately labelled.

## Frozen development-only check

`configs/topdial-debug-v2.json` freezes 64 development scenarios at ranks 16:80 of
`digest([17,id])`, disjoint from the original 16 calibration/selection scenarios.
Use root seed 17 and both profile/resistant protocols. Independently calibrate
revision-2 H_cover at the predefined .75 quantile on the original 16 prefixes;
H_edge comes from pre-test development trials. This is an expanded diagnostic,
not a replacement claim for the original frozen comparison budget or test lock.

GPU 0 runs PPDPP and the original IQE TDRL-R/Full. GPU 1 runs repaired MRNFixed/IQE
TDRL-R/Full. Expected counts are 384 and 512 dialogues (896 total). Fitting imports
are hash-audited; controller/runtime interfaces are checked. Both GPUs must pass
fresh BF16 dual-residency/controller, covered/off-map and exact journal/plan replay
smoke before either diagnostic evaluation begins. Record cold overhead separately.

Canonical command, inside allocation 58159 with one visible H100/eight CPUs:

```sh
runtime/.venv/bin/python scripts/debug_topdial_planner.py --slot 0 --mode smoke --output artifacts/topdial-debug-v2-58159/smoke-0
runtime/.venv/bin/python scripts/debug_topdial_planner.py --slot 0 --mode evaluate --output artifacts/topdial-debug-v2-58159/eval-0
```

Repeat slot 1 on the other GPU. Report actual progress/compute and scenario-paired
95% bootstrap CIs with 10,000 resamples; a single experimental seed provides no
training-variance evidence. Positive development results do not establish a win
on a pristine test set. A later official paired rerun requires a new global lock,
both-dataset selection and explicit post-test/exploratory disclosure.

## Namespace recovery

The initial worker stopped after 256 MRN dialogues, before its first IQE
assistant request, because the diagnostic omitted `request_namespace` while
using one shared generation journal for both heads. The transport rejected the
payload mismatch before HTTP dispatch; no pending request remains. Baseline
worker completed 384 dialogues. Preserve all original logs and manifests.

`diagnostic_contract` now namespaces the complete variant/head/checkpoint,
thresholds, code, config and target-node contract. Generation seeds still use
the paired scenario/protocol/turn context. Run fresh smokes and only the 256 IQE
dialogues in `artifacts/topdial-debug-v2-namespace-58159`; retained MRN records
are imported only after their partial-manifest hashes match. The final 896-row
report resolves to source hashes from retained and fresh segments. Do not resume
under the collided journal or rewrite the original manifests.

Canonical recovery: `python3 scripts/recover_topdial_debug.py`. After both fresh
runtime smokes, remaining IQE dialogue evaluation shares the allocation with
independent training-only numerical head refits. Numeric gates precede fitting;
new head checkpoints require a later calibrated dialogue evaluation. Imported
fitting audit checks unchanged executed inference/controller interfaces; the
numerical learner implementation is not executed by frozen-checkpoint dialogue
diagnostics. Keep BF16 forward, models, seeds, case IDs and prompts unchanged.
