# Bias & Calibration Audit — demo-biased

*Generated 2026-09-26 · 4800 responses · 1200 questions · 4 ordering(s) · 4 options · abstain/unparsable rate 0.0% · accuracy 56.5%*

## Summary

| Audit | Key statistic | Value | p-value | Verdict |
|---|---|---|---|---|
| Position bias | χ² GOF vs uniform (Cohen's w = 0.116) | χ² = 16.2 | 0.0010 | **moderate** |
| Answer-key balance (dataset) | χ² GOF vs uniform (w = 0.065) | χ² = 5.1 | 0.1679 | **none** |
| Length bias | χ² GOF by length rank (w = 0.197) | χ² = 46.5 | < 1e-6 | **moderate** |
| Length artifact (dataset) | gold-at-length-rank GOF | — | 0.6340 | **balanced** |
| Ordering consistency | same content across orderings | 0.551 [0.535, 0.569] | — | **unstable** |
| Ordering × correctness | McNemar on paired correctness | χ² = 23.2 | 1.4e-06 | **systematic direction** |
| Calibration | ECE (equal_width, 10 bins) | 0.190 [0.175, 0.204] | — | **poorly calibrated** |
| Selective prediction | AURC / E-AURC | 0.272 / 0.160 | — | — |

## 1. Position / label bias

Selection rates by presented position, with cluster-bootstrap 95% CIs (resampling questions, so re-ordered variants of one question move together):

| Position | Selection rate | 95% CI | Excess vs other positions |
|---|---|---|---|
| A | 0.234 | [0.221, 0.247] | -0.022 [-0.038, -0.006] |
| B | 0.226 | [0.215, 0.239] | -0.032 [-0.047, -0.016] |
| C | 0.321 | [0.307, 0.334] | +0.094 [+0.077, +0.112] |
| D | 0.219 | [0.208, 0.231] | -0.041 [-0.058, -0.026] |

χ²(3) = 16.2, p = 0.0010, Cohen's w = 0.116 → **moderate**. First-position selection rate: 0.234 (uniform would be 0.250).

![Selection rate by position](fig_position.png)

### Accuracy by gold position

If the model were position-blind, accuracy would not depend on where the correct answer sits (given a balanced key):

| Gold at | A | B | C | D |
|---|---|---|---|---|
| Accuracy | 0.644 | 0.614 | 0.579 | 0.605 |

Homogeneity χ² p = 0.4103.

## 2. Answer-key balance (dataset side)

| Gold at | A | B | C | D |
|---|---|---|---|---|
| Share of questions | 0.269 | 0.246 | 0.259 | 0.226 |

χ²(3) = 5.1, p = 0.1679 → **none**. An imbalanced key inflates position bias: a model with a matching slot preference scores better than it deserves.

## 3. Length bias

| Statistic | shortest | rank 1 | rank 2 | longest |
|---|---|---|---|---|
| Selection rate | 0.210 | 0.212 | 0.265 | 0.312 |
| Gold share (dataset) | 0.237 | 0.246 | 0.257 | 0.261 |

Selection by rank: χ²(3) = 46.5, p = < 1e-6, w = 0.197 → **moderate**. Mean length z-score of the selected option = +0.147 [+0.107, +0.189], one-sample t-test p = < 1e-6. Gold-at-rank balance: p = 0.6340 → **balanced**.

![Selection rate by length rank](fig_length.png)

## 4. Ordering consistency

Across 7200 ordered pairs from 1200 multi-variant questions: same-answer-content rate = **0.551** [0.535, 0.569] (per-question chance level ≈ 0.353); same-correctness rate = 0.698. McNemar χ² = 23.2, p = 1.4e-06 → **systematic direction** (e.g. canonical-order advantage).

## 5. Confidence calibration

Accuracy = 0.565, mean stated confidence = 0.755 → model is **overconfident**. ECE = **0.190** [0.175, 0.204], MCE = 0.240 → **poorly calibrated** (equal_width binning, 10 bins, n = 4800).

![Reliability diagram](fig_reliability.png)

## 6. Selective prediction (abstention)

Sorting answers by stated confidence: AURC = 0.272, E-AURC = 0.160.

**No useful operating point exists for risk ≤ 15%**: even the most confident 1.3% of answers still err at 14.1%. Confidence is too miscalibrated to threshold on (see §5) — recalibrate before deploying selective prediction.

![Risk-coverage curve](fig_risk_coverage.png)

## Recommendations

- Position bias detected (largest excess at **C**, +0.094). Report accuracy averaged over cyclic (or random) permutations of the options, or debias before comparing models.
- Length bias detected: control for option length (e.g. stratify accuracy by the gold option's length rank) or permute option order by length.
- Low ordering consistency: the reported score depends on which ordering you used. Average over orderings and report consistency alongside accuracy.
- Systematic correctness direction across orderings (often canonical-order memorization): treat single-order scores as optimistic.
- Calibration is overconfident: consider temperature scaling on held-out data. Confidence is too miscalibrated for thresholding — recalibrate first.

## Method notes

- All CIs are cluster bootstrap percentile intervals (B = 1000 by default) with questions as clusters.
- χ² and t-tests use one ordering per question: re-ordered variants of the same question are strongly correlated, and testing on all rows would overstate significance. Rates and CIs use every response.
- χ² tests use α = 0.01; effect size is Cohen's w (0.1 / 0.2 / 0.3 ≈ small / medium / large).
- E-AURC is the excess risk-coverage area over the oracle ordering (all correct answers first); smaller is better, 0 is unattainable in practice.
- Abstain/unparsable answers are excluded from position, length and calibration statistics and counted in the abstain rate.
- Equal-width bins can be noisy at the extremes; pass `--binning equal_mass` for quantile bins.
