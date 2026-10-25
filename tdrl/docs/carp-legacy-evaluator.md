# CARP legacy evaluator mode

`carp_legacy_v1` is an isolated reproduction of the public evaluator in
`/home/u26anhvl/crs/carp_www`. It does not replace the stricter CRS evaluator,
its semantic judge, or any existing report.

The mode pins the CARP source files and the two seed-2911 resources by SHA-256.
It evaluates 200 TopDial and 200 DuRecDial2.0 scenarios, shuffles each fixed
population with seed 42, starts every dialogue from an empty history, uses a
10-turn limit, and enforces the original DuRecDial minimum-success turn of two.
The user simulator and moderator receive the original plain-text prompts as one
`user` message at temperature zero. Their HTTP request has no seed, top-p,
top-k, or structured-output field. The moderator accepts `yes`/`no`, permits
one repair, and fails if the repair remains ambiguous. Refusal is not a terminal
condition. Reports contain SR, average turns, stop reasons, errors, per-domain
metrics, raw dialogue transcripts, moderator outputs, request journals, token
counts, latency, and GPU telemetry.

Baseline policy code is not rewritten. DirectLLM-v2 and ProCoT retain their
adapters; T-EPL-v2 and DPDP reuse the completed fitted checkpoints; EnPL-domain3
reuses its immutable training bank. Their internal assistant/search decoding
remains method-specific. Only the external CARP user simulator, moderator,
scenario population, initial state, and stopping protocol are standardized by
this mode.

This protocol is not interchangeable with the current fixed1000-prefix CRS
evaluation: it has a different 200+200 population and only one CARP user
protocol, so it cannot supply the CRS profile/resistant table or overwrite its
results.

Mini compatibility jobs are submitted in this dependency order:

1. DirectLLM-v2
2. T-EPL-v2
3. EnPL-domain3
4. ProCoT
5. DPDP

Each job requests two H100s and runs TopDial and DuRecDial2.0 concurrently, one
dataset per GPU. A later job starts only when the preceding job succeeds.

```bash
bash scripts/submit_carp_legacy_minis.sh
```

Outputs and scheduler logs are written below
`/mnt/data/u26anhvl/crs/carp_eval`. Mini outputs are diagnostic only and are not
scientific full-evaluation results.
