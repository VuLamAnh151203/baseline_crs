# PPDPP reward audit and local judge design

Follow-up: the user authorized implementation with "fix". The implemented v2
amendment and remaining H100/budget gates are recorded in
[PPDPP_FIDELITY_REPAIR.md](PPDPP_FIDELITY_REPAIR.md). Descriptions below of the old
active step and proposed design are the original audit snapshot, not current job status.

Audit date: 2026-10-07. This is source inspection and a CPU arithmetic/autograd
diagnostic, not a new inference experiment or evidence of improved success.
The active allocation 57711 step .2 still uses the existing frozen protocol;
no running source or reward contract was changed by this audit.

## Original mechanism

Inspected upstream commit `f44dccdd43f335e5e5f372ab556cfd9f182f9c55`:

- [env.py](https://github.com/dengyang17/PPDPP/blob/f44dccdd43f335e5e5f372ab556cfd9f182f9c55/env.py):
  `step` calls a critic after every assistant/user exchange. For ESC, the labels
  worse/same/better/solved map to -1/-0.5/+0.5/+1; CIMA uses
  incorrect/did-not-try/part/whole with the same values. `compute_reward` requests
  10 critic samples at temperature 1.1 and averages recognized ESC/CIMA labels;
  no recognized labels returns zero. ESC terminates successfully above 0.5,
  CIMA at 1. Bargaining uses deal detection and a negotiated-price reward,
  with different aggregation. These are original tasks, not CRS success rubrics.
- [prompt.py](https://github.com/dengyang17/PPDPP/blob/f44dccdd43f335e5e5f372ab556cfd9f182f9c55/prompt.py):
  the critic sees the dialogue and task and judges semantic progress; code parses
  its constrained answer, rather than matching acceptance expressions in raw user text.
- [agent.py](https://github.com/dengyang17/PPDPP/blob/f44dccdd43f335e5e5f372ab556cfd9f182f9c55/agent.py):
  categorical learned strategy policy, discounted returns normalized within each
  dialogue, summed REINFORCE loss, one update per completed episode.
- [run.py](https://github.com/dengyang17/PPDPP/blob/f44dccdd43f335e5e5f372ab556cfd9f182f9c55/run.py):
  optional SFT initialization; defaults are gamma 0.999, learning rate 1e-6,
  10 sampling rounds of 100 episodes. These are code defaults, not verified
  published experimental settings. There is no special replay buffer or PPO in
  this inspected RL loop.

## Current adaptation loses its terminal signal

### Paper versus repository: reward shaping must be qualified

The [paper, Section 3 and Appendix E.4](https://arxiv.org/html/2311.00262v2)
describes critic-derived terminal rewards and a small negative reward (example
-0.1) for ongoing interaction. In contrast, the pinned repository's `env.step`
returns `compute_reward` directly on every exchange; `run.train` appends that
reward without replacing ongoing ESC/CIMA feedback with -0.1. CB's -0.1 is
inside its no-deal branch, not a general additional step penalty.

Thus the paper describes a length penalty, and the repository has graded
goal-conditioned feedback, but these are not identical reward schedules. No
potential-based `gamma*Phi(next)-Phi(current)`, target-distance bonus, novelty
bonus, or entropy regularizer was found in the inspected repository RL path.
The recommendation to restore varying per-turn critic feedback follows the
**pinned code**. It must not be presented as an exact reproduction of the paper's
terminal/ongoing distinction. Our -0.05/+1 schedule matches neither fully.

`policy_training.py` uses -0.05 for every non-accepted turn and +1 for acceptance;
`ppdpp.py` retains upstream per-dialogue return normalization. For equal-length
trajectories whose only reward difference is the last turn, these return vectors
are positive affine transforms of one another:

`G_t = a/(1-gamma) + gamma**(T-1-t) * (b - a/(1-gamma))`.

Here `a=-0.05`, `gamma=0.99`, and terminal `b` is either -0.05 or +1. Both
coefficients of the exponential are positive, so mean/std normalization cancels
the terminal-reward difference except for epsilon and rounding. This does not
make every actual policy gradient identical: lengths, states and sampled actions
can differ, and one-turn dialogues bypass normalization. It does establish that
the terminal acceptance bonus barely changes the relative return weights for
fixed-length trajectories under this adaptation.

`artifacts/low-sr-audit/reward-normalization-diagnostic.json` contains an 8-turn
CPU arithmetic diagnostic and an autograd check using the actual
`tdrl.ppdpp.reinforce_loss`: maximum log-probability-gradient difference
4.76837158203125e-7. This is a reward-adaptation defect, not evidence that upstream
PPDPP cannot learn. Graded, varying critic rewards restore information that the
constant-step binary adaptation discards; success improvement remains untested.

## Wider fidelity audit

References below are to the same pinned repository. Local paths are relative to
`tdrl/`. Findings are source-level differences, not measured causes of poor SR.
`artifacts/low-sr-audit/ppdpp-source-audit.json` records inspected source SHA-256
hashes; source was downloaded for inspection only, never imported or executed.

| Area | Original repository / paper | Current adaptation | Assessment |
| --- | --- | --- | --- |
| Reward and termination | `env.py`: LLM critic, 10 samples, temperature 1.1; task-specific score and success threshold | `policy_training.py`: acceptance v1 yields +1, every other turn -0.05; no critic | Critical mechanism departure; binary bonus loses signal under retained normalization |
| Training case selection | `env.reset`: random `np.random.choice` from training | `policy_training.py`: sort IDs, deterministic prefix/cycle; skip initially refused states | Important undisclosed distribution departure; 256 episodes may cover only a sorted prefix, not random training cases |
| SFT supervision/actions | `data_reader.py`, `prompt.py`: provided strategy labels; task-specific 8 ESC / 5 CIMA / 11 CB actions | `official_data.py`: six regex weak labels, default acknowledge | Necessary CRS adaptation, but weak-label quality/skew requires evidence; not original gold strategy supervision |
| Action realization | `prompt.py`: each action maps to a full natural-language instruction | `evaluation.py`: append bare `Next strategy: <label>` | Weaker action interface; explicit per-action instruction mapping is absent |
| Policy input | `agent.build_input`: role-labelled recent turns, per-turn tokenizer separators; no explicit target/profile header | `evaluation.policy_input`: up to 96 header tokens plus recent turns, single outer CLS/SEP | Declared task-conditioning adaptation; exact tokenization/role serialization differs |
| SFT optimizer/schedule | `sft.py`: bias/LayerNorm no-decay groups; other weights decay .01; linear warmup/decay, default warmup 400 | `policy_training.py`: constant-LR torch AdamW, .01 decay for every parameter | Missing native scheduler and no-decay groups; material unlisted training departures |
| SFT checkpoint selection | `sft.py`: validation-selected best checkpoint | Local final-epoch checkpoint only; no per-epoch strategy validation | Material departure. Despite variable name `best_f1`, actual upstream criterion `results[0]` is macro precision, not F1 |
| RL optimizer | `agent.py`: transformers AdamW with only LR supplied | torch AdamW defaults: eps 1e-8, weight_decay .01 | Potentially material library/default and update-formula difference; see qualification below |
| RL clipping | `agent.optimize_model`: no explicit gradient clipping | `policy_training.py`: clip norm at 1.0 | Added optimization amendment, distinct from native SFT clipping |
| RL budget/hyperparameters | Paper Table 6 and code defaults: 1000 episodes, LR 1e-6, gamma .999 | 256 episodes, LR 6e-6, gamma .99 | Smaller budget and altered optimization; not native published settings |
| Shared dialogue protocol | Native tasks; paper: temperature 0, max 8 turns, 32 new tokens; code train temperature .7/test 0 | Both phases temperature .7, max 10 turns, 128 tokens; alternating profile/resistant in RL | Deliberate common-backbone/environment adaptation, still affects difficulty/variance and comparison to original evidence |
| Refusal and turn metrics | Native termination is goal or turn limit; actual completed length counted even on failure | Sticky refusal early stop, initial-refusal training filter, failed dialogues reported at full turn budget | Required local autonomy/evaluation protocol; disclose instead of claiming native environment equivalence |
| Policy model/runtime | Native default roberta-large; no explicit BF16 cast in inspected loading path | Pinned roberta-large BF16, gradient checkpointing and audited strict backward/replay | Required resource/reproducibility adaptation; numerical/update equivalence is not established by architecture alone |
| Auxiliary embedding | Native policy takes tokenized dialogue; no Qwen embedding request | `PPDPPPlanner.act` makes a live embedding request but discards its vector | Adds compute without informing PPDPP action; local common-runtime measurement departure |

HF AdamW in the [official Transformers v4.28.1 source](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/optimization.py)
defaults to eps 1e-6 and zero weight decay; its epsilon/bias-correction placement
also differs from PyTorch AdamW. The original PPDPP repository does not pin its
Transformers environment, so v4.28.1 is an illustrative verified implementation,
not proof of the precise package version used in published runs. Local optimizer
settings should be explicit, and update equivalence needs a numerical fixture
before claiming faithful migration between libraries.

Preserved mechanisms: pooled RoBERTa output, dropout 0.5, sorted categorical
actions, supervised cross entropy, training-time categorical sampling, evaluation
argmax, reverse-recursive discounted returns, per-dialogue sample-std
normalization and summed REINFORCE loss, one optimizer update per dialogue.
The upstream RL evaluator does not explicitly switch train/eval mode; its dropout
state depends on the call path. Local `policy.eval()` is deliberate stabilization,
not an asserted exact equivalence to every original execution path. No PPO,
learned value baseline, replay buffer or entropy term was found in the inspected
RL update.

The generated fidelity artifact's `status: validated` means local fitting checks
completed. It does not establish native PPDPP training/evaluation equivalence.
Reward, sampling and optimizer differences need to be reflected explicitly in
future fidelity manifests, rather than relying on the current four-item
adaptation list. Prioritize reward and case sampling, then optimizer/SFT selection
and action instructions. Model/backbone and refusal changes remain governed by
the local common protocol, not by literal upstream copying.

## Proposed protocol amendment, not implemented

Use the pinned BF16, non-thinking Qwen/Qwen3.5-9B already resident in the same
H100 allocation. Semantic judgments replace raw-utterance regex decisions;
JSON/schema parsing only validates transport output and must not decide intent.

Keep two separate outputs:

1. **Outcome judge:** task, dialogue prefix and permitted profile/knowledge;
   method identities, distances, planner traces and existing parser labels hidden.
   Recommendation outcomes distinguish explicit unpressured target acceptance,
   rejection, mention, unresolved and uncertainty. Resolve pronouns/paraphrases
   from context, require cited user-turn evidence, and assess refusal/pressure.
   Prior consumption and assistant assertions are not new user acceptance.
   Non-recommendation DuRecDial goals require explicit task-specific success
   definitions before evaluation; do not equate all goal completion with acceptance.
2. **Training critic:** graded progress after each real assistant/user exchange,
   separately versioned from terminal success. Develop CRS-specific rubrics;
   upstream ESC/CIMA labels cannot be copied literally. Preserve native policy
   and REINFORCE mechanics when restoring per-turn critic feedback. Quasimetric
   Bellman targets retain positive reachability step cost, independent of critic reward.

Freeze rubric, model revision, prompt hash, decoding, sample count/aggregation,
seeds, evidence schema and uncertainty/error handling on training/development.
Ten samples per turn follow upstream sampling structure but add material compute;
fewer samples are a disclosed adaptation that requires measured development
stability. Invalid output is an infrastructure error, not success or silent regex
fallback. Judge uncertainty is retained explicitly, never dropped from denominators.
Record all judge calls/tokens/time and exact replay IDs. Apply the same outcome
judge to PPDPP, TDRL-R and TDRL and both simulation protocols.

The latest user request permits discussing replacement of the earlier
deterministic-primary rule; any implemented replacement needs an explicit
versioned amendment to AGENTS/config/manifests. Preserve old artifacts as a
different protocol. Resume neither old binary-reward RL checkpoints nor old
selection metrics under new contracts; audit reuse of SFT, embeddings and heads
individually, rerun affected fitting/smoke/development/selection, then lock before
test. Offline rejudging cannot reconstruct training rewards, termination or
dialogue continuations that would have changed under the new judge.

Qwen serving as assistant, simulator and judge remains same-family simulation
evidence. It is not independent validation and does not replace the required
approved blinded human study.
