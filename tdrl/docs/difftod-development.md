DiffTOD development baseline for the active TopDial debugging goal
================================================================

The baseline uses the completed 100001-update native diffusion policy's EMA checkpoint selected by minimum likelihood loss on development excluding the frozen confirmation families. The shared Qwen3.5-9B realization, live Qwen3-Embedding-8B, simulator, local semantic outcome judge, root seed 17 and both protocols remain unchanged. Discovery scenarios are the same digest-sorted development ranks 16:80 used by the MRN planner diagnostics. This command does not open confirmation or official-test bodies.

After fitting completes, run the following as separate exclusive one-GPU/eight-CPU Slurm steps in allocation 58159, from a clean committed checkout with the locked runtime and required deterministic/offline environment. Run slot 0 and slot 1 smokes on distinct GPU UUIDs; retain both outputs.

```sh
runtime/.venv/bin/python scripts/evaluate_difftod_development.py --fitting artifacts/difftod-topdial-fit-v1-58159 --mode smoke --slot 0 --output artifacts/difftod-development-v1-58159/smoke-0
runtime/.venv/bin/python scripts/evaluate_difftod_development.py --fitting artifacts/difftod-topdial-fit-v1-58159 --mode smoke --slot 1 --output artifacts/difftod-development-v1-58159/smoke-1
```

Each smoke executes two full-budget training-case dialogues and completed diffusion-plan/journal replay, verifies 5125 policy forwards per five-candidate plan, and measures a conservative full-turn proxy against the original allocation clock. The reserve covers the existing scoped matrix; it does not authorize official testing or replace measurements for the other methods. A failed feasibility gate requires an explicit scope/budget decision.

```sh
runtime/.venv/bin/python scripts/evaluate_difftod_development.py --fitting artifacts/difftod-topdial-fit-v1-58159 --mode evaluate --slot 0 --gates artifacts/difftod-development-v1-58159/smoke-0 artifacts/difftod-development-v1-58159/smoke-1 --output artifacts/difftod-development-v1-58159/discovery
```

The evaluation produces 128 per-dialogue records and protocol aggregates. This remains single-seed development evidence. Compare paired scenario records against MRNFixed TDRL-R/Full with matching realization/simulator/judge contracts; freeze the selected planner before disjoint confirmation. Neither passing tests nor the generated development aggregates prove superiority.

After the two MRN workers and DiffTOD discovery baseline complete, generate the paired report from their hashed records:

```sh
runtime/.venv/bin/python scripts/compare_difftod_development.py --difftod artifacts/difftod-development-v1-58159/discovery --mrn artifacts/topdial-semantic-strategy-debug-v8-58159/eval-0 artifacts/topdial-semantic-strategy-debug-v8-58159/eval-1 --output artifacts/topdial-semantic-strategy-debug-v8-58159/paired-difftod
```

The report requires complete development manifests, identical scenario order/population, pinned model and shared prompt/serialization provenance, complete paired cells, matching judge/turn budgets and actual role-specific model/decoding records. It rejects unresolved infrastructure errors and non-finite/inconsistent compute. Per-protocol bootstrap results use 10,000 resamples; the macro-average clusters both protocols by scenario. These are discovery diagnostics, not proof from disjoint confirmation. A positive point estimate alone does not establish improvement.

Warm-dialogue feasibility extrapolates **all** measured per-turn work, including
intrinsic diffusion sampling, to the shared full unsuccessful turn limit. An
early success never leaves denoising work unscaled. The corrected post-fit gate
reserves the original 3664 matrix dialogues plus 256 for the second planner
candidate. Previous preflight projections remain historical artifacts; they are
not substituted for the corrected fitted-policy gate. Method-specific R/Full
smoke timings must also pass remaining-clock feasibility before matrix launch.

Completed DiffTOD live embeddings now have a scoped, versioned journal. Each new
real turn still performs a live full 4096-D embedding. Replaying the identical
completed request returns its exact normalized vectors without another HTTP call;
request text/order, model revision/weights, instruction, preprocessing and adapter
hash must match. Corrupt, mismatched or unknown pending outcomes fail closed.
This closes an embedding duplication exposed by the fitted-policy replay gate;
old failed smoke artifacts remain preserved. The diffusion sampler, selected EMA,
semantic prompts, decoding, response policy and 5125-forward budget are unchanged.
