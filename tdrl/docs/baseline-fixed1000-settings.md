# Shared fixed-test settings for baselines

`configs/baseline-fixed1000-test-v1.json` is an opt-in evaluation overlay for
PPDPP/TDRL, DiffTOD, TRIP, and DPDP. It leaves each method's training recipe,
checkpoint format, existing comparison config, and prior reports unchanged.

All methods keep using the full official train and development splits. The test
set is the committed, hash-verified resource already used by the fixed1000
TDRL/DiffTOD runner: 500 TopDial and 500 DuRecDial2.0 scenarios. `--fixed-test-count
400` selects the first 200 from each dataset; `--fixed-test-count 1000` selects
all 500 from each. The execution-order contract is first 200 TopDial, first 200
DuRecDial2.0, then the remaining 300 from each dataset. Evaluation uses seed 17,
natural budget, `profile` and `resistant` protocols, and a 10-turn limit.

The fixed subset is test-only. It does not change model training data or dev
selection. Seed 17 and natural-only apply to official-test evaluation; each
method keeps its existing development-selection matrix (including natural and
matched where required) and its native training settings. The method-specific
dev run and selection lock must be completed before opening test bodies. Since
this implementation changes evaluator code, create fresh dev evidence and
selection locks from a clean committed checkout before test evaluation. Keep
the method-specific checkpoints and natural/matched configs required by the
existing selection workflow.

For the core TDRL/PPDPP comparison, pass the overlay when starting the normal
comparison pipeline:

```sh
PYTHONPATH=src runtime/.venv/bin/python -m tdrl.cli comparison \
  --config configs/comparison-judge-v2-sample500-seed17.json \
  --output /mnt/data/duclm3/baseline-fixed1000/tdrl \
  --fixed-test-setting configs/baseline-fixed1000-test-v1.json \
  --fixed-test-count 1000
```

The core end-to-end comparison is a one-shot pipeline. Use the final count of
1000 for that command; use a separate fresh output root for a 400-case initial
run if needed.

For the standalone baseline evaluators, add these options to their existing
`evaluate --split test` command (after generating fresh dev evidence and the
corresponding selection lock):

```sh
--fixed-test-setting configs/baseline-fixed1000-test-v1.json \
--fixed-test-count 1000
```

Supported commands are `tdrl.baselines.dpdp.evaluate_cli`,
`tdrl.baselines.trip.evaluate_cli`, and `tdrl.baselines.difftod.evaluate_cli`.
Use their existing checkpoint, model-lock, encoder/tokenizer, plan-store, and
selection-lock arguments; this overlay does not replace those artifacts.

Test outputs live under `test-fixed1000/`, separately from existing full-test
outputs, and record the setting hash, frozen-resource hash, selected IDs and
content hashes. Standalone reports are count-specific, so a 400-case report is
not overwritten by a later 1000-case report. Dialogue request IDs do not
include the prefix count, allowing the 400-case prefix to be reused when the
same method, checkpoint, selection, and output root are extended to 1000. Keep
separate output roots if changing any of those inputs. The frozen execution
order defines the sample prefix and recorded ID order; method runners may group
execution by dataset because each method loads dataset-specific checkpoints.

This setting is a sampled official-test evaluation, not full-test or human-study
evidence. The code changes only establish a shared data/evaluation contract;
they do not establish that any scientific run has completed or passed H100
feasibility gates.
