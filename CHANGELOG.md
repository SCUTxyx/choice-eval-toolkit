# Changelog

## 0.2.0 — 2026-09-26

Research-grade pass over statistics, robustness and reproducibility.

### Statistics
- **Variant-permutation test** for directional ordering asymmetry. The exact
  McNemar test pools C(V,2) correlated pairs when a question is asked under
  more than two orderings, which is anti-conservative; the new test permutes
  variant labels within each question (label-exchangeability null), is valid
  for any number of orderings, and reduces to exact McNemar for two.
- **Confidence AUROC** (discrimination) added to the calibration audit with
  cluster-bootstrap CI — calibration and discrimination are independent
  properties, and the report now measures both.
- **Vectorized cluster bootstrap**: one resampling pass per audit instead of
  one per position/rank, and cluster grouping rewritten from O(rows × clusters)
  to O(rows log rows) — large runs are orders of magnitude faster.
- Accuracy-by-gold-position homogeneity test now guards degenerate tables
  (0% / 100% accuracy runs).
- Permutation p-values at the Monte-Carlo floor are rendered as `p < x`
  rather than `p = x`.

### Robustness
- Missing inputs degrade to clearly marked *skipped* sections instead of
  crashes: no `confidence` (calibration + selective prediction), no
  `option_lengths` (length audit), no `option_ids` (ordering audit),
  all-abstain runs (selection statistics; the dataset-side audits still run).
- Duplicate `(question_id, variant_id)` records are rejected with a clear
  error instead of silently double-counting.
- One-option-count-per-run is enforced with an explicit message (was implicit).

### Reproducibility & outputs
- Every report records the toolkit version, bootstrap/permutation counts,
  binning, seed, α and target risk, and ships a `results.json` next to it.
- New `Accuracy by ordering` table in multi-ordering runs.
- `bundle_to_dict` / `write_results_json` for programmatic pipelines.

## 0.1.0 — 2026-09-26

Initial release: position/label bias, answer-key balance, length bias and
gold-length artifact, ordering consistency, ECE/MCE calibration, selective
prediction; synthetic generator with injectable known biases; 33-test
injection-recovery suite; CLI (`audit`/`demo`) and Markdown reports.
