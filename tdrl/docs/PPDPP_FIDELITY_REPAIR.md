# PPDPP fidelity repair: protocol v2

Implementation amendment authorized by the user on 2026-10-07. Scientific success
improvement is not established by the code or CPU tests. Keep all old artifacts;
the prior step stopped at a durable checkpoint (TopDial 256 updates, DuRecDial 92).
Use allocation 57711 with its original start time and expiry. No new allocation.

```bash
python3 tdrl/scripts/run_in_allocation.py -- \
  runtime/.venv/bin/python -m tdrl repair-ppdpp \
  --source artifacts/low-sr-fix-57711 \
  --output artifacts/ppdpp-judge-v2-57711
```

The source inspection is in [PPDPP_REWARD_AUDIT.md](PPDPP_REWARD_AUDIT.md).

The initial v2 attempt passed dual/controller smoke but stopped during budget
preflight before any fitting/update: a judge cited a non-existent history index.
Its recommendation fixture also exposed a goal-completion/acceptance label mix.
Retain that entire attempt. The correction uses numbered history, a goal-specific
label schema and a constant observed-user index. The application attaches the
exact observed message as the quote, avoiding generated evidence text altogether.
Generation finish_reason is supplied to the judge, with a truncated ambiguous
commitment regression fixture. No old request is reissued or reused. The corrected
fresh output directory is `artifacts/ppdpp-judge-v2b-57711` (replace `--output` above).
Same-code/config/source restarts can resume completed stages; unknown pending
requests or changed contracts stop for audit. A failed budget manifest never
authorizes fitting.

The second attempt stopped during sampled-critic smoke, again with zero fitting
updates: independent output fields permitted a refusal with unchanged progress.
The final transport schema enumerates 17 internally valid structured decisions
for the declared goal and observed user index. XGrammar compilation was verified
without inference before the next H100 smoke. The model still chooses the semantic
decision; no contradictory sample is repaired, dropped or replaced. Retain both
failed attempts; use fresh `artifacts/ppdpp-judge-v2c-57711` for this correction.

Audit of the third attempt's real development traces found false successes on
different songs, music instead of the target restaurant, and already playing a
song without a new choice. Retain but invalidate its budget projection as a
calibration estimate under a defective topic criterion; do not claim its 8/8
proxy successes as results. The next protocol explicitly separates target_act
and target_topic and asks the model to resolve selected_topic first (canonical
target, OTHER_TOPIC, NO_NEW_CHOICE or UNCERTAIN). Success is structurally possible
only for the canonical target. Negative decisions retain graded feedback. The
65 valid structured combinations still forbid contradictory flags/reward.
New H100 regression cases cover alternative songs, off-goal activities and prior
consumption. Use fresh `artifacts/ppdpp-judge-v2d-57711`; no fitting or request
outcomes from prior attempts are reused. Same-family semantic accuracy still
requires independent evaluation; structural constraints are not that evidence.

The fourth attempt failed its mention fixture before fitting. The next calibration
decides new_choice first, with explicit mention/conditional/prior-consumption
counterexamples, before topic resolution. Private profiles are removed from
acceptance judge input; permitted knowledge is supplied only for information
goals. The seven-field finite schema does not permit success without new_choice.
Retain the fourth attempt; use `artifacts/ppdpp-judge-v2e-57711` for the new smoke.

The fifth attempt passed mention/conditional checks but guessed a target choice
from a truncated sentence. The corrected schema excludes success for all
finish_reason=length observations, without interpreting text by regex. Shared
assistant and simulator prompts request one or two short sentences (native PPDPP
also uses concise response instructions); all methods retain the same 128-token
budget. Partial observations still receive sampled non-terminal graded feedback.
Use fresh `artifacts/ppdpp-judge-v2f-57711`; old fixtures/projections remain retained.
The reference is pinned code `f44dccdd43f335e5e5f372ab556cfd9f182f9c55`, which
returns graded critic feedback per exchange. The paper's ongoing length penalty
and terminal reward are a different schedule; this repair follows the code.
There is no added potential, distance, novelty or entropy shaping.

## Restored training mechanics

- Sample uniformly with replacement from all permitted training scenarios using
  a fixed NumPy seed. Persist the complete episode-to-scenario schedule; do not
  take a sorted prefix or remove initially refused scenarios.
- Restore ten sampled Qwen critic judgments per training turn at temperature 1.1,
  without top-k truncation. Map worse/unchanged/progress/achieved to
  -1/-0.5/+0.5/+1 and average samples. No fixed -0.05 step penalty or binary +1
  override. Preserve normalized discounted-return REINFORCE, one update per
  episode, no added RL clipping, 1000 episodes, LR 1e-6, gamma .999.
- Use explicit legacy HF AdamW update math/defaults in RL; no weight decay,
  epsilon 1e-6. During SFT, epsilon 1e-8, weight decay .01 excluding biases and
  LayerNorm weights, 400-step warmup and linear decay. Match the native first
  update's schedule factor while using a resumable modern scheduler.
- Retain BF16 forward parameters; keep optimizer master weights and moments in
  FP32 on the same H100 so tiny updates accumulate rather than vanish in rounding.
  This is an explicit numerical amendment, not bitwise native FP32-forward equivalence.
- Select the SFT checkpoint by macro precision on frozen official development
  strategy examples, matching the actual upstream selection code (its `best_f1`
  variable name is misleading). Keep a separate resume checkpoint, optimizer,
  scheduler/RNG state, per-epoch confusion matrices, and selected epoch.
- Realize every strategy through a full instruction rather than a bare label.

## Semantic outcome and common protocol

All methods use one deterministic, non-thinking, pinned Qwen local outcome call
after each real user response. This decision handles contextual pronouns and
paraphrases; it requires exact quoted user evidence and schema consistency.
Recommendation acceptance must cite the latest user response. Refusal ends the
dialogue, pressure cannot count as success, and uncertainty remains unsuccessful
in the full denominator. JSON-schema constrained decoding is transport validation,
not regex inference about user intent; invalid output stops as an infrastructure
error without a parser fallback.

Ten stochastic training-critic calls are separate from the common outcome call.
This separates training feedback from a deterministic success/termination rubric.
It is a disclosed departure from the original critic's dual role. Store every
sample, score, prompt/config hash, seed/request ID and call/token/time counter.
Judge input hides methods, planner traces, distances and old parser labels; it
contains only the current target, permitted profile/knowledge and observed prefix.

Information/question goals require task completion observed in dialogue; they
are not counted as recommendation acceptance. Output reports include goal type,
recommendation-acceptance count and other-goal-completion count. All denominators
and official split membership remain intact. Offline trajectory ends without
independent success evidence are censored, never accepted by regex. Directed
reachability retains its independent positive step-cost objective and declared
censored negative objective; it does not use critic reward as distance.

Keep required shared Qwen/BF16 backbone, profile/resistant simulation, role
decoding, 10-turn/128-token budget, full-budget unsuccessful-turn convention and
live embeddings. Six weak CRS strategy labels, bounded target/profile header,
local token serialization, and evaluation dropout disabled are disclosed task
adaptations. No human labels are fabricated or collected. Weak-label validation
scores do not establish gold strategy quality. Same-family Qwen simulation/judging
does not establish real-user performance or replace approved human evaluation.

## Ordered gates and budget

Fresh Gates 0/1/2, then judge/controller smoke precede fitting. Judge smoke checks
explicit acceptance, pronouns, mention, conditional interest and refusal; ten-sample
critic journal replay; real live embedding; both BF16 Qwen models resident alongside
RoBERTa and all FP32 optimizer state; exact repeated actions/log-probabilities/gradients
with actual gradient checkpointing. The trained SFT policy repeats the replay gate
before any RL update. Reduced vLLM memory reservations (.48 generation/.36 embedding)
leave room for FP32 optimizer state; precision/model/context length are unchanged.
Dual-residency failure stops the runtime gate, without substitution/offloading.

After Gate 2, eight real development proxy dialogues estimate latency and native
RL/complete scoped test cost. Public corpus totals minus audited original train/dev
counts project sealed test counts (final membership is verified only post-lock).
The projection uses all 24 unique seed/protocol/method-budget cells per scenario,
safety factor 1.25, a fitting reserve and the original allocation clock. Timing is
from an untrained intrinsic policy, not a method result; small-sample p95 and uniform
method-cost assumptions are explicit limitations. Method-specific post-fit budget
checks remain necessary. If the proxy does not fit, retain its projection and stop
before full fitting for an explicit scope/budget decision; do not silently shrink
seeds, splits, methods, protocols, critic sampling or RL episodes.

If these gates pass, refit embeddings, both heads, SFT and RL under fresh manifests,
run PPDPP development diagnostics, then fresh selection/test/report. No fitting or
selection is reused across the changed contracts. Original source/checkpoint hashes
and zero pending requests are audited before branching. The allocation holder's
stop signal remains connected. Test bodies stay sealed until the new selection lock.
