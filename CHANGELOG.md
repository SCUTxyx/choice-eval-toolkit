# Changelog

## 0.3.0 — 2026-09-26

Adversarial self-audit round: two confirmed correctness/validity defects found
by constructing failing inputs, plus the fix for each.

### Fixed
- **Abstention threshold was not deployable under tied confidences.** The
  suggested cut was chosen as an arbitrary prefix boundary, which can fall
  *inside* a group of tied confidences (typical when confidence saturates at
  0.99). The reported `(threshold, coverage, risk)` then described a prefix
  that no deployable `conf >= t` rule could reproduce — demonstrated with an
  adversarial case where the true rule risk was 2× the reported one. The
  threshold is now selected among tie-consistent cuts (maximum coverage with
  rule risk within target), and a regression test replays the rule on the data
  and asserts the stated numbers exactly.
- **`expected_offset_rates` base share** (introduced with this release,
  caught by its own recovery test before shipping).

### Added
- **Gold-offset position test (confound-free).** The marginal slot test
  assumes a balanced answer key: a position-*blind* model on an imbalanced
  key with decent accuracy was flagged "severe" (demonstrated: χ² = 2025 on a
  bias-free synthetic model). The new test conditions on being wrong —
  `(selected − gold) mod K` must be uniform over the K−1 offsets regardless of
  key balance and accuracy — and the two tests are reported together with a
  documented coverage argument. Combined verdict introduces **inconclusive**
  for "marginal signal + imbalanced key + silent offset test" instead of
  over-claiming.
- Generator knob `offset_attract` (relative-position preference) with
  closed-form ground truth `expected_offset_rates`; a fifth row in the README
  recovery table.
- Chunked bootstrap resampling so very large runs don't materialize a
  B×clusters index matrix.

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
