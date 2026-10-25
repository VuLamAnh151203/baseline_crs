# EPL source-corrected v2

The completed fixed-400 run `59830` is retained as immutable diagnostic evidence,
but it is not the paper-facing EPL result. Its adapter selected the final root
action by visit count. With five rollouts over five policy candidates, equal visit
counts deterministically selected action index zero; 3,226 of 3,281 evaluated
turns therefore selected `acknowledge`.

Version 2 (`source-corrected-v2`) preserves the released DuRecDial command's
five P-UCT rollouts, depth five, retrieval top-k 20, BERT policy encoder,
MiniLM retriever and train/dev-only memory. It corrects the execution contract
to match the pinned EPL source:

- chance nodes include the source's initial zero return;
- P-UCT uses `sqrt(log(parent visits))` and the source default exploration
  constant 100;
- the final root action maximizes backed-up mean sampled return, not visit count;
- simulated accept/refuse text does not terminate the internal tree; `farewell`
  or the search horizon does;
- tree responses remain simulations, while the selected action receives a fresh
  final realization before entering the real dialogue.

The TopDial/DuRecDial action taxonomy and shared Qwen realization, simulator,
judge, seed, protocols and fixed test membership remain disclosed dataset/runtime
adaptations. The corrected mini is diagnostic only and never opens official test
bodies.

```bash
bash scripts/submit_epl_corrected_mini.sh
```
