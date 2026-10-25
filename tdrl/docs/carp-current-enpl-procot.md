# EnPL and ProCoT under the current CARP evaluator

This integration evaluates the native `EnPL-domain3` and `ProCoT` planners
without copying or modifying CARP's simulator, moderator, stopping, metrics, or
quality judge.  The external CARP checkout is pinned to commit
`e5bf7034d7526bcc9cf6869666e8bebc80d36cc0`; evaluator source files and both
200-scenario resources are hash-checked before any dialogue is opened.

The adapter translates only CARP's typed scenario and observed history into the
baseline planner contract.  EnPL retains its verified train-only plan bank,
domain/entity overlap retrieval (`k=3`), history-conditioned two-call planner,
and native decoding.  ProCoT retains its native prompt, parser, one-call planner,
and native decoding.  Neither method is retrained.  CARP owns the empty initial
history, user simulator, acceptance moderator, maximum ten turns, minimum
success turn (TopDial 1, DuRecDial2.0 2), and the separate 0--3 Ovr. pass.

Dialogue generation uses the CARP batch-invariant Qwen3.5-9B BF16 server
(vLLM 0.30 plus PR #60033, 16k context, 256 sequences, 32768 batched tokens).
The published CARP quality protocol is preserved by restarting the same model
on the stock vLLM server for Ovr. scoring.  Results are resumable by scenario;
native assistant calls also use a provenance-bound request journal.

Submit the validated chain from `tdrl/`:

```bash
bash scripts/submit_carp_current_enpl_procot.sh
```

The chain is CPU gate -> two-scenario-per-dataset mini for both methods -> full
200-scenario-per-dataset evaluation for both methods.  The full job is submitted
immediately with an `afterok` dependency so it accrues queue priority while the
gates run.  Its immutable output root is:

```text
/mnt/data/u26anhvl/crs/carp_eval/carp-current-e5bf703-full-v1
```

`report.json` is written only after all four result sets contain the requested
unique scenario count, zero evaluator errors, and complete Ovr. judgments.
EnPL and ProCoT do not define CARP's retrieval-based covered/OOD variable; the
adapter therefore records `coverage_available=false` and these runs must not be
used for a CARP covered-vs-OOD split.
