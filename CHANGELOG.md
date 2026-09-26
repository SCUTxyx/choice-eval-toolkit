# Changelog

## 0.5.0 — 2026-09-26

Embodied-evaluation extension: arena-style pairwise preference audits.

### Added
- **Pairwise preference audit** (`choice_eval/pairwise.py`) for
  RoboArena-style A/B judging of trajectories/plans/policies:
  - *slot preference* — P(judge picks what is shown first) vs coin flip,
    exact binomial test + pair-level cluster bootstrap CI, with a
    *slot-balance* design check that flags confounded (non-randomized)
    presentations — mirroring the MCQ answer-key balance audit;
  - *swap consistency* — same winner across both presentation orders of the
    same pair (order-sensitive judging, independent of slot balance);
  - *length preference* — P(chosen description is longer) + mean advantage;
  - *corrected leaderboard* — per-content win rates split by as-first /
    as-second slot.
- New schema: `PairJudgment` / `PairwiseRun`, `load_pairwise_jsonl`,
  `save_pairwise_jsonl`, and `load_any` (auto-detects MCQ vs pairwise from
  the first record; `choice-eval audit` handles both transparently).
- `generate_pairwise` with three closed-form ground truths
  (`expected_pair_slot_rate = 0.5 + s/2`,
  `expected_swap_consistency = (1-f)(1+d²)/2 + f/2`,
  `expected_p_chosen_longer = 0.5 + (1-d)·λ`); the README recovery table
  gains a pairwise section.
- `choice-eval demo-pairwise` — clean vs biased arena-style demo reports
  (`examples/pairwise_clean/`, `examples/pairwise_biased/`).
- **Risk ladder** in the abstention audit: deployable (threshold, coverage,
  achieved-risk) operating points at 5/10/15/20/30% risk — for embodied and
  other asymmetric-cost deployments where the right target is a policy
  choice.
- `docs/EMBODIED.md` — mapping guide from embodied evaluation formats
  (action-selection MCQ, embodied QA, trajectory preference arenas) to the
  two schemas, with per-format audit guidance.

### Fixed (caught by the new closed-form tests before release)
- Length-preference statistic misaligned winners with slots: the winner's
  content must be resolved through `slot_of_a`, which alternates across
  presentation orders — the slot-0 shortcut washed the signal out
  (0.496 measured vs 0.575 designed).
- Slot-balance check passed the success count as the trial count
  (`binomtest(k=sum, n=sum)` → p ≈ 0), flagging every balanced design as
  imbalanced.
- Variable shadowing in the risk ladder (`risk` reused) crashed MCQ audits.

## 0.4.0 — 2026-09-26

Public-beta hardening: strict data-quality gates at the loader, sharper
verdict semantics, and a structural-invariant fuzz suite.

### Added
- **Strict numeric typing in the JSONL loader.** Fractional indices
  (`gold_index: 2.5` was silently truncated — moving the answer key),
  booleans (`confidence: true` read as 100%) and stringly-typed numbers are
  rejected with a line-precise message instead of coerced. Non-object lines
  are rejected; UTF-8 BOM is tolerated.
- **Structural-invariant fuzz suite** (`tests/test_invariants.py`): eight
  randomized generator configurations hammered through the full pipeline
  asserting rates sum to 1, every CI covers its point estimate, no NaN leaks
  into JSON, and every configuration renders a report.
- Verdict semantics corner: under a *balanced* key both position tests are
  valid, so the combined verdict takes the more severe of the two (previously
  a strong marginal signal could be down-graded by a milder offset result).
  Documented with targeted tests.
- Slot labels for runs with more than 26 options (A…Z, then L27, L28, …).
- Performance reference: 100k responses audit end to end in ~30 s at B = 1000.

## 0.3.1 — 2026-09-26

Edge-case hardening round, found by inspection + targeted probes.

### Fixed
- **`equal_mass` binning with constant confidence silently reported ECE = 0.**
  When every response carries the same confidence (saturated models, unparsable
  answers backfilled with a constant), quantile binning degenerates to zero
  bins and the audit claimed "well calibrated" on data whose equal-width ECE
  was 0.139. Now falls back to a single bin over [0, 1] (ECE = |acc − conf|).
- **Bootstrap draw-shape ambiguity**: a length-1 vector statistic (k = 2 runs)
  was indistinguishable from a scalar and crashed the vector-CI helper. The
  draws array now keeps its shape; scalar/vector callers disambiguate.
- Latent variable coupling between the position and length figures.

### Added
- Input validation: `n_bins ≥ 1`, `n_boot ≥ 1`, `n_perm ≥ 1`,
  `0 < target_risk ≤ 1`, and `n_options ≥ 2` (one-option logs rejected with a
  clear message; k = 2 true/false-style runs verified end to end).
- Small-sample caution banner in reports with < 30 questions.
- `pip install git+https://...` quick-install line; project URLs in metadata.

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
