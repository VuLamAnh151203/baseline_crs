# Baseline adaptation contracts

Status: **PPDPP mechanism implemented and CPU-tested; H100 fitting/validation pending.
DPDP/TRIP/DiffTOD remain source-pinned adaptation designs.** References are official
author repositories; their old metrics/checkpoints/traces do not constitute new evidence.
The immutable source revisions are in `configs/sources.lock.json`.

PPDPP protocol v2 restores sampled graded feedback, uniform case sampling, explicit
legacy HF AdamW math and native SFT/RL settings. The common outcome protocol now
uses the user-authorized local semantic judge. See
[PPDPP_FIDELITY_REPAIR.md](PPDPP_FIDELITY_REPAIR.md) for remaining task adaptations,
fresh H100 gates, and budget preflight; completed fitting alone does not establish
native-task equivalence or improved success.

| Method | Mechanism to preserve | Adaptation and fidelity evidence required |
|---|---|---|
| [PPDPP](https://github.com/dengyang17/PPDPP) | Learned plug-and-play policy and RL policy optimization | Define training-only action taxonomy/target-acceptance reward for both datasets; train the intrinsic policy, preserve policy update/objective; realize its chosen action through locked Qwen; verify policy logits/action mapping/reward separation |
| [DPDP](https://github.com/cs-holder/DPDP) | Fast learned policy plus deliberative MCTS; offline RL followed by search-enhanced learning | Preserve fast/search dispatch, tree backup and training phases; substitute loopback Qwen only for LLM interfaces; retrain dataset policy and freeze online-update protocol before test; verify node calls, reward/transition semantics and budget exhaustion |
| [TRIP](https://github.com/iwangjian/TRIP) | Forward/backward target-constrained action/topic planning with bidirectional agreement | Preserve both planner decoders, consistency/contrastive training and constrained decoding; use the planned action/topic path as Qwen guidance; disclose replacement of original dialogue generator as a same-backbone adaptation, verify no future test goal-path leakage |
| [DiffTOD](https://github.com/ninglab/DiffTOD) | Diffusion trajectory planning with conditional guidance | Preserve SEDD-style diffusion policy training/sampling and trajectory guidance/selection; retrain on allowed trajectories, adapt DuRecDial input/actions, replace hosted realization/simulation/moderation with loopback Qwen; test denoising/guidance/resource counters and dynamic behavior |

These descriptions follow the source mechanisms, not an assertion that they already work
on both datasets under this resource protocol. The PPDPP source README at the pinned revision
contains only its title; consult the original [paper](https://arxiv.org/abs/2311.00262) and code.
DPDP README fetch at the recorded pin returned 404; this is a documentation-access issue, not
evidence that its implementation is absent. Inspect its source and [paper](https://arxiv.org/abs/2406.05374)
before writing its adapter. TRIP/DiffTOD README files were inspected directly from the pinned
author sources. DiffTOD's source evaluation uses hosted clients; those entrypoints must never
be imported or executed here.

Intrinsic learned policy models are permitted by AGENTS.md and are distinct from the common
language realization backbone. Pin their weights/architecture if used; they must fit on the
same allocated H100 during their evaluated stage. This is not permission to substitute the
required generation or embedding model. Baseline policy training may temporarily unload both
Qwen servers within the same allocation; interactive evaluation still requires both resident.

Every validated adaptation must have: upstream revision/license; local implementation hash;
architecture/checkpoint/input/action/reward provenance; only training fitting and development
selection; equivalence/fidelity tests against the native policy behavior on small fixtures;
identical allowed information; shared realization and user prompts; full resource trace;
natural/matched configurations; declared departures and limitations. Until those artifacts
exist, selection.lock is rejected. Prompt-only substitutes must be named as separate controls
and cannot fill these four required baseline slots.

Excluded scope: GDP-Zero and SCOPE are not required baseline slots under the user's updated
list; comparisons using other LLM realization/planning backbones are excluded. Historical
cross-backbone results cannot be placed in new comparison tables.


The current three-method job and explicit PPDPP departures are documented in
[COMPARISON.md](COMPARISON.md). No real adaptation is marked validated until its new
supervised and on-policy training artifacts complete.
