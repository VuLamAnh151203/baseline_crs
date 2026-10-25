# Low-success audit and development repair

Audit: 2026-10-07. Sources: allocation 57711, source commit 8a1e31a,
`artifacts/selection-fix-57711/selection`, and the completed earlier PPDPP RL stages.
This document separates bug fixes from expected performance; no improvement is yet
claimed. Official test bodies remain unopened.

## Observations

`artifacts/low-sr-audit/inspection.json` records all 256 development dialogues.
Training guidance carried source people/items/facts into unrelated current scenarios.
For example, current-target film and restaurant dialogues switched to unrelated
reference items. New graphs retain the observed training strategy, not the foreign
utterance's entity/factual content. Nodes, directed distances, edges, goals, budgets
and training membership remain unchanged.

The simulator sometimes treated profile Accepted/Rejected fields as an exhaustive
catalogue and invented contradictory prior consumption across turns. Shared assistant
and user prompts now distinguish training strategy from current facts and historical
preferences from a whitelist. Refusal, skepticism, topic shifts and correction remain
allowed; acceptance is never compelled. No stronger claim of simulator validity is
made without fresh traces/human evidence.

PPDPP's old RL training succeeded on 5/256 TopDial episodes and 3/256 DuRecDial episodes.
These are training diagnostics, not development/test baseline results. New shared
prompts affect its on-policy training, so both RL stages restart from the unchanged
SFT checkpoints with original seeds, 256-episode budgets, optimizer and reward fields.
Completed old RL artifacts are retained but not reused in the repaired comparison.
After fresh RL, PPDPP development diagnostics use the same 16 fixed scenarios per
dataset, seed 17, and both profile/resistant protocols before head selection.

## Smaller MRN/IQE engineering

Both heads expose `encode(states)` and `distance_encoded(source,target)`;
`distance(source,target)` composes these. Existing parameter names, dimensions,
checkpoint layouts and formulas remain intact: MRNFixed's Euclidean plus positive
max residual, IQE's directed interval-union/maxmean construction. No additional
package, custom kernel, approximation, new architecture or cosine substitution.

Graphs encode training references once and cache directed shortest paths per goal
set. Live states are still embedded and encoded at every turn/leaf. Head-content
mutation checks remain; edge updates invalidate path caches. This removes repeated
reference MLP evaluation, full reference-vector transfers and reverse graph traversal.
CPU tests compare values and gradients with the original formulas and preserve graph
orientation. An H100 compatibility stage compares all four trained heads in BF16
with original formulas and invariant tolerances before cached fitting is reused.
It also verifies every cached training terminal mask under the repaired refusal rule.
A mismatch stops reuse rather than silently mixing training targets.

## Protocol and selection fixes

Acceptance v1 remains conservative: mentions, pronouns, paraphrases and unmatched
commitment forms are not newly promoted to success. Refusal v2 additionally recognizes
clear forms such as "not really interested" and not needing further recommendations;
it remains sticky and can reduce measured success. Both methods share this policy.

A zero-success head has undefined compute per success (`null`), not a fabricated
finite value. With a CI containing zero, an observed-success head has finite cost
while the zero-success head's cost is unbounded; choose the finite-cost candidate.
Success counts must explicitly substantiate the undefined metric. If every valid
head has zero successes, stop and retain decision inputs. Other tie-breakers remain
unchanged. Recovered selection locks use validated schema v3 with mandatory recovery
provenance; v1/v2 continue to validate their original schemas, unknown versions fail.

## Data limitation retained

The current DuRecDial adapter includes non-recommendation goals such as asking about
hobbies and weather notification. One of the 16 fixed development scenarios has a
hobbies goal; the training corpus includes 1002 hobbies and 522 weather targets.
The current exact-target recommendation-acceptance rule is a poor fit for these goals.
No scenario, split or denominator is removed. Defining applicability or an alternative
task-success protocol requires a separate explicit scope decision, followed by
appropriate preprocessing/fitting/evaluation invalidation. Exact title/commitment
matching also remains a conservative measurement limitation. Neither limitation is
resolved by quasimetric refactoring or by increasing generation pressure.

## Run and provenance

Use only the existing one-H100 allocation:

```bash
python3 tdrl/scripts/run_in_allocation.py -- \
  runtime/.venv/bin/python -m tdrl repair-development \
  --source artifacts/selection-fix-57711 --output artifacts/low-sr-fix-57711
```

Verify immutable input/checkpoint hashes and frozen embedding/head/SFT contracts.
Retain the original allocation start time and stop marker. Rerun current-code Gates
0/1/2, cached-head compatibility, both PPDPP RL stages, PPDPP development diagnostics,
then fresh head selection/test/report. Old selection caches/traces/metrics are not
imported. Prompt/strategy/refusal changes and parent artifacts are recorded in the
new amendment; Gate 4 must freeze them before any test. Estimates are not results.
Human evaluation and the broader baseline/control matrix remain outside this scoped
job, with all institutional prerequisites unchanged.
