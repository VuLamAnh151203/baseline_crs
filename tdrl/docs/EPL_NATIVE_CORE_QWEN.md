# EPL native-core Qwen adaptation

`EPL-native-core-Qwen-fixed400` is the paper-facing, task-adapted EPL baseline.
It is isolated from the earlier `EPL` v1/v2 namespaces and does not overwrite
their code or artifacts.

The adaptation retains the released BERT-base-cased policy architecture
(768 -> 128 -> action logits), five epochs, optimizer schedule, goal-topic
action structure, MiniLM experience retrieval, and P-UCT settings (five
simulations, depth five, `cpuct=100`, top five policy actions). Language
realization, the interactive user simulator, and the semantic outcome judge use
the same locked local Qwen3.5-9B runtime as the other fixed evaluation methods.
Both committed and simulated goal-topic paths enter the policy serialization at
each P-UCT node, matching EPL's `[PATH]` state rather than using an empty path.

DuRecDial policy goals come from the release's turn-aligned
`goal_type_list`. TopDial has no native goal annotations, so it uses a frozen,
deterministic adapter based on the declared task target and observed assistant
turn only. To make the goal-topic classifier valid across datasets and unseen
entities, lexical topics are represented as `TARGET`, `CONTEXT`, or `NONE` and
resolved from visible state during realization. This is an explicit task
adaptation, not an unmodified reproduction of the EPL paper.

The diagnostic mini uses 32 policy-training examples, eight policy-development
examples, one train-only self-simulation scenario, and one development dialogue
per protocol and dataset. TopDial and DuRecDial run concurrently on distinct
H100s. It records native action distributions and exact completed-dialogue
replay, and never opens official test bodies.

Submit from a clean committed `tdrl` checkout:

```bash
bash scripts/submit_epl_native_core_mini.sh
```

Artifacts are written under
`/mnt/data/u26anhvl/crs/epl_native/epl-native-core-mini-e2e-JOB_ID`.
Submission or queueing is not evidence; the mini passes only when
`e2e-summary-mini.json` and `diagnostic.lock.json` both report completion.
