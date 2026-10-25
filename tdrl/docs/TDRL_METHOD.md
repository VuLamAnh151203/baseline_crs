# TDRL: Code-Accurate Method Description

> **Naming note.** In the current repository, **TDRL** provisionally means
> **Temporal Distance Reinforcement Learning**. This is a temporary method name,
> not a finalized paper title.
>
> **Scope note.** This document describes the implementation currently used by
> the repository. It distinguishes the shared TDRL mechanism from the two
> dataset-specific frozen variants used in the latest fixed-prefix evaluation.

## 1. Motivation

Target-driven conversational recommendation requires the system to move a live
dialogue toward a specified conversational goal without blindly copying a fixed
plan. A training dialogue may provide a useful next-move pattern when the live
state resembles a state from which the goal was historically reachable. The same
guidance can be misleading when the live dialogue lies outside the supported
region of the training map.

TDRL therefore separates planning into two modes:

1. **Fast, retrieval-guided planning** for states supported by a training-built
   directed map.
2. **Slow, rollout-based recovery** for unsupported states, where several next
   moves are simulated and the branch that returns closest to a supported route
   is selected.

The dispatch decision is made using a learned **directed temporal distance**. The
same directed distance is used to construct the offline graph, retrieve references,
define coverage, and score rollout leaves. This shared geometry is the central
mechanism of TDRL.

## 2. Method overview

```text
Training dialogues
      |
      v
Observed-prefix states --frozen Qwen embeddings--> normalized 4096-D vectors
      |                                                 |
      |                                                 v
      +-- temporal transitions/goals --> directed quasimetric head
                                                        |
                                                        v
                                      training-only directed guidance graph

Live dialogue history
      |
      v
4096-D live embedding --> directed retrieval --> covered by H_cover?
                                                /                    \
                                             yes                      no
                                              |                        |
                                  retrieved strategy/path       candidate rollouts
                                              |                        |
                                              +------> final response <+
```

At inference time, TDRL never adds hidden future turns, success labels, test
outcomes, or target metadata to the state embedding. The embedding represents only
the dialogue history observed at that moment. The target, profile, and permitted
knowledge are supplied separately to response realization and goal-conditioned
map construction.

## 3. Dialogue-state representation

Let the observed dialogue state at turn $t$ be

\[
s_t = (m_1, m_2, \ldots, m_t),
\]

where each $m_i$ is an observed user or assistant message. The serialization
contains only the ordered history:

```json
{
  "version": 1,
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

It deliberately excludes:

- the target and user profile;
- future dialogue turns;
- success, reward, and split labels;
- test membership or dataset markers.

The frozen `Qwen/Qwen3-Embedding-8B` encoder maps this serialization to a full
4096-dimensional vector. The runtime uses the instruction:

> Represent the observed dialogue state for directed reachability retrieval.

The resulting vector is L2-normalized. Truncation, malformed dimensions,
non-finite values, zero norms, model substitution, and unstable source order are
rejected rather than silently repaired.

## 4. Learning directed temporal distance

### 4.1 Why direction matters

Conversational reachability is asymmetric. A state near goal completion may be
easy to reach from an earlier state, while reversing that progression is not the
same planning problem. TDRL models an ordered cost

\[
d(s, g) \neq d(g, s),
\]

where a smaller $d(s,g)$ means that goal state $g$ is estimated to require
less remaining conversational progress from state $s$.

The implementation supports two quasimetric heads over frozen state embeddings:

- **MRNFixed:** a symmetric Euclidean component plus a directed positive-residual
  component;
- **IQE-maxmean:** a learned interval-quasimetric representation whose component
  costs are combined by a learned convex maximum/mean mixture.

Both heads are constructed to support non-negativity, zero self-distance, and the
triangle inequality. Their asymmetry and numerical invariants are checked after
training. A cosine distance is retained only as a symmetric control, not as the
primary TDRL geometry.

### 4.2 Offline transition construction

Training dialogues are converted into trajectories of observed-prefix states.
Each actionable state occurs immediately before an observed assistant move. Its
successor is the next observed conversational state in the same trajectory. The
final observed state is retained, but under the current `censored_v2` protocol an
observed dialogue ending is not automatically treated as successful goal
achievement.

Reachable goals are sampled from the same legitimate future portion of a training
trajectory. Cross-trajectory goals and censored/failed terminals are used only for
negative separation; they are not falsely labeled as reachable one-step goals.
In the current sampler, 75% of draws attempt a within-trajectory reachable goal
and 25% draw a cross-trajectory censored contrast.

### 4.3 Bellman-style temporal objective

For a reachable transition $s_t \rightarrow s_{t+1}$ and sampled goal $g$, the
detached target is

\[
y(s_t,g) =
\begin{cases}
0, & s_t = g,\\
c_{\text{step}} + \gamma\,
\mathbf{1}[s_{t+1} \neq g]\,
d_{\text{target}}(s_{t+1},g), & \text{otherwise}.
\end{cases}
\]

The reachable loss is a smooth-L1 regression loss between the online distance
and this target. Censored contrasts use a margin-separation objective:

\[
\mathcal{L}_{\text{neg}}
= \lambda_{\text{neg}}
\max(0, m_{\text{neg}} - d(s,g)).
\]

The current common training configuration is:

| Parameter | Value |
| --- | ---: |
| Optimizer steps per head | 30,000 |
| Batch size | 256 |
| Learning rate | $3 \times 10^{-4}$ |
| $c_{\text{step}}$ | 1.0 |
| $\gamma$ | 1.0 |
| Negative margin | 10.0 |
| Negative weight | 1.0 |
| Target-network EMA $\tau$ | 0.01 |
| Forward precision | BF16 |

The selected TopDial numerical revision retains BF16 model/input/forward
computation but uses FP32 auxiliary optimizer and target-EMA master values. The
online head, detached target head, optimizer state, RNG state, sampler state, and
checkpoint provenance are resumable. The currently evaluated DuRecDial2.0 IQE
head is retained from the earlier globally selected run and is not silently
reinterpreted as a refit under this TopDial numerical revision.

This is the “reinforcement learning” component in the current TDRL core: temporal
distance is learned through a Bellman-style bootstrapped objective over offline
transitions. TDRL does **not** fine-tune the Qwen generation or embedding models,
and it does not train its planner through online policy-gradient interaction.

## 5. Training-only directed guidance map

After fitting the distance head, TDRL creates reference nodes from training
states only. A reference contains:

- the observed dialogue-prefix state;
- its frozen 4096-D embedding;
- the observed next assistant move or its strategy abstraction;
- the training dialogue ID and target metadata.

For two reference nodes $u$ and $v$, TDRL records the directed edge

\[
u \rightarrow v \quad \text{iff} \quad d(u,v) \leq H_{\text{edge}},
\]

with edge weight $d(u,v)$. The implementation never infers a reverse edge from
an existing forward edge. Directed shortest paths are computed toward training
goal anchors.

Goal anchors are reference states whose observed next assistant move mentions the
declared training target. This target mention is used only to identify a possible
route endpoint; it is not interpreted as user acceptance or dialogue success.

`H_edge` controls graph connectivity. It is distinct from the inference-time
coverage threshold `H_cover`; the two thresholds are calibrated separately and
then frozen before test evaluation.

## 6. Directed retrieval and coverage

For a live state $s$, TDRL embeds the observed history and considers only
training references that have a recorded directed path to a goal anchor. It
computes the ordered live-to-reference cost

\[
d(s,r),
\]

not $d(r,s)$. Let $r^*$ be the selected reference. The live state is
retrieval-covered when

\[
\operatorname{covered}(s)
= \mathbf{1}\left[d(s,r^*) \leq H_{\text{cover}}\right].
\]

Coverage therefore means **support under the frozen training-built retrieval
map**. It must not be described as universal in-distribution/out-of-distribution
classification or as an intervention-invariant prediction of failure.

## 7. Fast path: retrieval-guided realization

When the state is covered, TDRL uses the selected training reference to choose a
conversational strategy and asks the shared local Qwen generator to realize that
strategy using only the current conversation, current target, profile, and
permitted knowledge.

The shared strategy space is:

1. `acknowledge`;
2. `ask_preference`;
3. `clarify`;
4. `explain`;
5. `recommend`;
6. `farewell`.

The current TopDial v12 variant can also provide the observed training next move
as a form-and-intent example. The prompt explicitly forbids copying reference
people, preferences, item attributes, or factual claims. It additionally extracts
only current knowledge triples whose subject matches the current target and
reminds the generator to preserve subject bindings and acknowledge missing facts.

This path normally requires one live-state embedding and one final assistant
generation. It is represented by the `TDRL-R` ablation when recovery is disabled
for all states.

## 8. Slow path: conditional rollout recovery

Full TDRL activates recovery when the state is not covered. Under the current
natural budget it uses three candidate branches and a lookahead depth of two.
Each branch performs the following operations:

1. Generate a candidate next assistant response from a strategy instruction.
2. Simulate the next user response with the frozen local user-simulation prompt.
3. Optionally judge whether the simulated user has accepted, refused, or been
   pressured.
4. If the branch is still active and depth remains, generate the next assistant
   continuation and simulate another user response.
5. Embed the resulting leaf dialogue state.

For planner revision 2, a non-terminal leaf $z_i$ is scored by its cheapest
directed return to the guidance map:

\[
J_i = \min_{r \in \mathcal{R}_{\text{goal}}}
\left[d(z_i,r) + C_{\text{path}}(r \rightarrow G)\right],
\]

where $C_{\text{path}}$ is the sum of recorded forward-edge costs from the
reference to a goal anchor. Semantically accepted branches receive score zero;
refused or pressured branches receive infinite cost. TDRL selects the finite
branch with minimum cost, with deterministic candidate-index tie-breaking.

The first TopDial v12 candidate preserves the retrieval-guided response as a
control candidate. The other branches explore alternative strategies. Thus slow
recovery asks whether a short simulated intervention can move an unsupported live
state back toward a region for which training-derived guidance exists.

The local simulator and judge do not access method identity, learned distances,
candidate-selection status, or hidden future test turns. Nevertheless, rollout
quality is bounded by the same-family Qwen simulation and judgment protocol; it
is not evidence of real-user behavior.

## 9. Full TDRL versus TDRL-R

| Component | TDRL-R | Full TDRL |
| --- | --- | --- |
| Frozen state embedding | Yes | Yes |
| Directed graph and retrieval | Yes | Yes |
| `H_cover` measured/logged | Yes | Yes |
| Covered-state guidance | Yes | Yes |
| Off-map candidate rollouts | No | Yes |
| Leaf-to-map directed scoring | No | Yes |
| Conditional fast/slow routing | No | Yes |

TDRL-R isolates the value of the learned representation and retrieval map. Full
TDRL adds the adaptive-compute hypothesis: spend additional inference only when
the learned map does not sufficiently support the current state.

## 10. Current frozen dataset-specific variants

The latest fixed-prefix evaluation does not use one identical planner revision on
both datasets. The exact frozen variants are:

| Property | TopDial | DuRecDial2.0 |
| --- | --- | --- |
| Directed head | MRNFixed | IQE |
| Planner revision | 2 | 1 |
| Map scope | Exact target when supported; otherwise goal-type transfer | Goal-type transfer |
| Strategy labels | Complete semantic overlay over the training bank | Original training strategy labels |
| Ambiguous guidance | Excluded | Legacy behavior |
| Farewell guidance | Excluded | Legacy behavior |
| Observed next-move example | Enabled | Disabled |
| Initial reference selection | Nearest directed entry | Legacy entry/path behavior |
| Retrieval candidate preserved in recovery | Yes | No |
| Semantic rollout terminals | Yes | No; explicit-refusal terminal behavior |
| Leaf score | Entry distance + remaining directed path cost | Entry distance only |
| Current-target fact grounding | Enabled | Disabled |

TopDial uses the user-selected **v12 MRNFixed Full** candidate frozen before its
disjoint development confirmation. DuRecDial2.0 retains the earlier globally
selected **IQE Full** model, original thresholds, training bank, and planner
revision 1. Therefore, results should be reported as dataset-specific frozen TDRL
implementations. The DuRecDial run must not be described as a v12/MRN refit.

## 11. TopDial v12 strategy overlay

The original TopDial bank contains weak next-action labels. The v12 pipeline adds
a frozen semantic annotation over every actionable training state. The annotator
sees only:

- the declared training target;
- the preceding observed history;
- the observed next assistant move.

It does not see learned distances, future user turns, outcomes, development/test
records, or the old weak label when making the semantic decision. The resulting
label replaces only the strategy annotation; the source state, embedding, target,
bank order, and distance checkpoint remain unchanged. Ambiguous and farewell
references are excluded in v12.

This overlay is part of the frozen TopDial method implementation, not additional
test-time supervision.

## 12. Development selection and locking

The general training pipeline fits MRNFixed and IQE with the same state bank,
sampler, optimizer budget, and seed. Each candidate must pass quasimetric,
stability, and provenance checks. `H_edge` and `H_cover` are calibrated separately
using training/development resources.

The general head-selection rule is:

1. Remove candidates that fail invariants, stability, or provenance.
2. Prefer the higher macro development success rate.
3. If the paired 95% CI contains zero, prefer lower H100 seconds per success.
4. If compute differs by less than 1%, prefer fewer parameters.
5. Break a remaining tie in favor of MRNFixed.

The post-debug TopDial v12 candidate follows an additional frozen discovery and
disjoint-development-confirmation protocol. Before official test data are opened,
the runner writes an immutable lock that pins the code commit, package lock,
models, checkpoints, graph bank, strategy overlay, thresholds, prompts, dataset
archives, fixed scenario resource, and confirmation evidence. Changed or missing
inputs cause a hard failure; there is no hidden fallback or test-time reselection.

## 13. Shared local models and decoding

TDRL uses:

- `Qwen/Qwen3.5-9B` for assistant realization, user simulation, semantic rollout
  judgment, and final outcome judgment;
- `Qwen/Qwen3-Embedding-8B` for normalized 4096-D dialogue-state embeddings;
- BF16 inference on H100 GPUs inside Slurm;
- local loopback-only vLLM transport with thinking disabled;
- no hosted inference and no external API fallback.

For the current fixed-prefix experiment, assistant and simulator decoding both use
temperature 0.7, top-p 0.8, top-k 20, and at most 128 generated tokens. The
deterministic outcome judge uses temperature 0 and a schema-constrained response.

## 14. Dialogue evaluation protocol

Each scenario exposes the same initial user observation, target, profile,
personality, and permitted knowledge to every compared method. A dialogue runs for
at most ten assistant turns under two frozen protocols:

- **profile:** the simulated user follows the supplied profile but may disagree,
  correct, refuse, or change topic;
- **resistant:** the user additionally challenges weak claims and resists
  unsolicited persuasion.

After each assistant response, the shared simulator generates a user reply. A
separate structured semantic judgment determines whether the latest user turn
demonstrates a new, unpressured choice of the exact target or completion of the
declared information goal. Mention, politeness, conditional interest, prior
consumption, selection of another item, refusal, pressure, and truncated replies
do not count as success.

The dialogue ends on success, explicit refusal, or the ten-turn limit. For the
reported average-turn metric, unsuccessful dialogues are charged the full turn
budget even if refusal ends the interaction early; measured inference compute
continues to use the actual executed calls and elapsed time.

## 15. Current fixed-400 and extension protocol

The amended test resource freezes 500 TopDial and 500 DuRecDial2.0 scenario IDs.
Runtime code never resamples or substitutes cases. Execution order is:

1. 200 TopDial;
2. 200 DuRecDial2.0;
3. the remaining 300 TopDial;
4. the remaining 300 DuRecDial2.0.

Consequently:

- `--count 400` evaluates the first 200 scenarios from each dataset;
- `--count 1000` extends the same immutable run to 500 scenarios per dataset.

Both profile and resistant protocols use experimental root seed 17 and natural
planner budgets. TDRL alone therefore produces 800 dialogues at count 400 and
2,000 dialogues at count 1000. A paired TDRL-versus-DiffTOD comparison produces
1,600 and 4,000 dialogues, respectively.

Stopping count and GPU slot do not enter dialogue request identities. Completed
dialogues have checksum seals and durable request journals, allowing the 400-case
prefix to be extended or an interrupted run to resume without changing or
recomputing completed evidence.

## 16. Metrics and statistical reporting

The implementation records at least:

- success rate;
- average turns;
- H100 seconds per dialogue and per success;
- assistant/simulator/judge generation calls;
- embedding calls;
- input, output, and total generated tokens;
- recovery triggers, selected candidates, paths, and rollout scores;
- per-turn latency, request IDs, seeds, prompt/model hashes, and replay evidence;
- refusal, pressure, and semantic outcome labels.

Paired method comparisons use 10,000 scenario-cluster bootstrap resamples with
seed 20261003. Profile and resistant observations belonging to the same scenario
remain paired. These intervals quantify scenario-sampling uncertainty for the
frozen single-seed experiment; they do not quantify training-seed variability or
real-user uncertainty.

## 17. What the method may and may not claim

The current implementation supports the following bounded interpretation:

- TDRL learns a directed temporal geometry from training-only dialogue
  transitions.
- It uses retrieval-defined support to choose between low-cost guidance and
  higher-cost recovery.
- Full TDRL can be compared with TDRL-R to isolate the value and cost of recovery.
- Evaluation measures reproducible local interactive simulation under frozen
  profile and resistant protocols.

It does **not** establish that:

- coverage is a universal OOD detector;
- learned distance causally predicts success under every intervention;
- Qwen-simulated users represent real users;
- a target mention is acceptance;
- the same final TDRL revision was fitted on both datasets;
- single-seed fixed-prefix results measure training-seed uncertainty;
- dialogue quality or human preference follows from success rate alone.

## 18. Implementation map

| Concern | Main implementation |
| --- | --- |
| State serialization | `src/tdrl/state.py` |
| Local generation and embeddings | `src/tdrl/backend.py` |
| MRNFixed and IQE heads | `src/tdrl/reachability.py` |
| Bellman/separation learning | `src/tdrl/learning.py` |
| Offline bank and head fitting | `src/tdrl/training.py` |
| Directed graph and retrieval | `src/tdrl/graph.py` |
| Conditional fast/slow planner | `src/tdrl/planner.py` |
| Semantic strategy overlay | `src/tdrl/strategy_annotations.py` |
| User simulation and dialogue loop | `src/tdrl/evaluation.py` |
| Semantic outcome/rollout judge | `src/tdrl/judge.py` |
| Selection lock | `src/tdrl/selection.py`, `src/tdrl/confirmation_lock.py` |
| Metrics and paired bootstrap | `src/tdrl/metrics.py` |
| Frozen fixed-prefix membership | `src/tdrl/test_prefix.py` |
| Current fixed-prefix worker | `scripts/evaluate_fixed1000_prefix.py` |
| Current fixed-prefix queue | `scripts/run_fixed1000_prefix.py` |
| TopDial v12 configuration | `configs/topdial-semantic-strategy-debug-v12.json` |
| Current paired-prefix configuration | `configs/fixed1000-prefix-tdrl-difftod-v1.json` |

## 19. Compact paper-style summary

TDRL learns a directed quasimetric over frozen dialogue-state embeddings to
estimate the remaining temporal cost from an observed conversational state to a
training-supported goal state. Training-only states form a directed graph whose
edges, live-state retrieval, coverage decision, and rollout-leaf scores all use
the same ordered distance. At inference, a covered state follows retrieved
strategy guidance through a low-cost realization path. An uncovered state
activates a higher-cost recovery procedure that generates several candidate next
moves, simulates short user interactions, and selects the branch with minimum
directed cost back to a goal-reaching region, while rejecting refused or pressured
branches. Separate development-frozen thresholds control graph connectivity and
fast/slow dispatch. The generator and embedding models remain frozen; only the
lightweight directed-distance head is learned. Current evidence is based on local
single-seed Qwen simulation and must be interpreted as simulated conversational
planning performance rather than real-user deployment performance.
