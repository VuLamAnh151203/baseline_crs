# EnPL independent reconstruction and TDRL adaptation

This implementation is an independent reconstruction of *Thoughts to Target: Enhance
Planning for Target-driven Conversation* (EMNLP 2024). The public ConvPlan repository contains
the released candidate data but no executable EnPL implementation. Therefore neither this code
nor its future measurements should be described as author-code results or an exact reproduction.

The supplied PDF has SHA256 `334e11ef...922b9`, which differs byte-for-byte from the hash quoted
inside the engineering guide. The supplied 17-page PDF is the authority used for prompt extraction;
its full hash is pinned in `configs/enpl-reference.lock.json`.

## Source-to-code ledger

| Paper source | Preserved mechanism | Implementation |
|---|---|---|
| Section 3.1.1, Appendix A.1.3 | Natural-language plans and distillation template | `schema.py`, `bank.py`, `templates/paper_distill.txt` |
| Section 3.1.2, Eqs. 1–3 | TextRank entity sequences, normalized Levenshtein, top 2 of 10 | `entity_extractor.py`, `consistency.py`, `bank.py` |
| Section 3.2 | Similarity equals movie-set overlap | `normalize.py`, `retrieval.py`; generalized to same-domain entities as an explicit adaptation |
| Section 5.4, Figure 3 | Three similarity demonstrations | `retrieval.py`, frozen config |
| Appendix A.2.2 | New-plan prompt | `templates/paper_generate_plan.txt` |
| Appendix A.3.2 | Per-turn plan-guided response | `templates/paper_generate_response.txt`, `planner.py` |
| Appendix E, Table 8 | Reported 12,147-plan corpus statistic | `audit.json` discrepancy field |

PDF line-wrap hyphenation was removed, typographic apostrophes were normalized, and template
placeholders were converted to Python format fields. The adapted templates are separate files and
do not overwrite the paper-reference templates.

## Frozen decisions

| Setting | Value | Provenance |
|---|---|---|
| Candidates per released record | 10 | [P] Section 3.1.2 |
| Candidates retained | 2 | [P] Section 3.1.2 |
| Consistency | phrase-token Levenshtein, normalized by maximum length | [P] Eqs. 1–3 |
| Extractor | deterministic token-graph TextRank, window 4, damping 0.85 | [D] author implementation unavailable |
| Keyword budget/order | 20, first occurrence, keep first duplicate | [D] guide default |
| Demonstration similarity | same-domain exact entity-overlap cardinality | [A] movie overlap generalized without embeddings |
| Demonstrations | 3, distinct source conversations | [P] count; [D] distinctness/ties |
| Zero-overlap behavior | consistency-ranked deterministic fill within the same domain, explicitly logged | [A] |
| Online behavior | regenerate plan every assistant turn | [P] Appendix A.3.2 |
| Planner context | observed history plus shared visible profile/knowledge | [A] TDRL integration |
| Context overflow | prefer 3 examples; drop lowest-ranked whole examples above 12,000 input characters, then fail closed | [D] |
| Backbone | shared Qwen3.5-9B BF16 | [A] controlled comparison |
| Evaluation | fixed-400, seed 17, natural budget, profile/resistant, 10 turns | [A] shared protocol |
| Historical model/API settings | unavailable | [U] |

The runtime makes two policy calls per executed assistant turn: one `planner` call and one
`assistant` response call. It performs no embedding request. The planner has a baseline-local
output policy and must return a numbered 2--8 step plan; malformed plans fail closed. Generated
plan text is retained in the turn trace but is never appended to the real dialogue history.

The shared evaluation contains movie, music, food, POI, celebrity/news, hobby and generic goals.
The adapted retriever maps each target to a frozen coarse domain, extracts exact preference
entities from matching profile fields, and retrieves only same-domain training plans. It preserves
the paper's cardinality score, three demonstrations and consistency tie-break, but it is not the
paper's movie-only retriever. This adaptation is frozen before official tests.

## Leakage and bank policy

The raw ZIP is never modified. Released conversations are exact-matched, after deterministic
normalization, against the already prepared official DuRecDial train and development scenarios.
Only uniquely matched training records with canonical targets are eligible. Development,
ambiguous, and unmatched records remain visible in audit artifacts but cannot enter a scored
bank. Official test bodies remain sealed until after the method lock.

The release contains 5,541 records and 10 candidates per record. Keeping two per record gives at
most 11,082 candidates before provenance exclusions, while Table 8 reports 12,147 plans. The
implementation records this discrepancy and never pads or relabels the released data.

The completed CPU audit (Slurm job 59869) found 5,538 uniquely matched training records, two
ambiguous released IDs (`423`, `4786`) sharing one normalized conversation, and one unmatched ID
(`4526`). Two candidate plans were empty. The verified bank therefore contains 11,076 entries;
the complete audit is stored under `/mnt/data/u26anhvl/crs/enpl/enpl-bank-audit-59869`.

## Execution

CPU-only validation, submitted from the project root:

```bash
./scripts/submit_enpl_cpu_gate.sh
```

Build and audit the complete released bank on a CPU worker without opening official tests:

```bash
./scripts/submit_enpl_bank_audit.sh
```

After committing a clean checkout, run the two-GPU diagnostic. It builds the complete bank,
evaluates one development scenario per dataset under both protocols, freezes the method lock,
and performs real request-journal replay smoke tests. It does not open official tests.

```bash
./scripts/submit_enpl_mini.sh
```

Only after the mini passes, run fixed-400 (200 scenarios per dataset, two protocols, 800
dialogues):

```bash
./scripts/submit_enpl400.sh
```

Outputs live under `/mnt/data/u26anhvl/crs/enpl`. Every dialogue retains raw generation journals,
the generated plan, selected demonstration IDs and groups, overlap scores, template hashes,
latencies, token usage, response-format diagnostics, simulator turns, and semantic-judge output.

The publication label should be **EnPL-domain3 (independent reconstruction, adapted)**.
