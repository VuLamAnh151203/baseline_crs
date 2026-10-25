# Data provenance

Raw data and prepared corpora are gitignored. Official dataset references are pinned in
`configs/sources.lock.json`:

- [TopDial](https://github.com/iwangjian/TopDial): official repository links its curated dataset
  on Google Drive. Repository license alone is insufficient to assert the downloaded corpus
  license; preserve and audit the actual release terms and split membership.
- [DuRecDial 2.0](https://github.com/liuzeming01/DuRecDial): official documentation specifies
  CC BY-NC-SA 4.0 for the corpus and Apache 2.0 for code. Audit the downloaded release, languages
  and bilingual grouping to avoid train/dev/test translations crossing membership.

Do not treat a repurposed legacy corpus as an official split without an explicit documented
membership audit. Source schemas, scenario counts, language policy, truncation and action/topic
taxonomy remain pending. Test input opening is blocked until immutable selection verification.
Near-duplicate auditing of locked test bodies must be post-selection and may invalidate the
paired evaluation; it must not tune training or thresholds from test content.

`tdrl prepare` consumes already adapted canonical JSONL with exact keys:
`id`, `messages`, `profile`, `knowledge`, `target`. It requires audited source provenance,
official split membership hash, license, input/adapter hashes and transformation list. It
exports only observed prefixes into the embedding-state field and preserves training next
assistant moves as a separate guidance field. Its manifest stays partial until the cross-split
audit is completed. No official raw-data adapter has yet been implemented.
