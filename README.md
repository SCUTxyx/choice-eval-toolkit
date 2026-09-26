# choice-eval-toolkit

**Audit multiple-choice evaluation runs for position/label bias, length bias and ordering
instability — and check whether the model's stated confidence actually means anything.**

Multiple-choice (MCQ) evaluation is quietly fragile. Models develop slot preferences (a
model that over-picks **C**), grab the longest option, or answer differently when the same
options are shuffled — and a benchmark whose answer key is itself imbalanced makes all of
this worse. A single accuracy number hides all of it. This toolkit takes an evaluation log
and tells you, with hypothesis tests and confidence intervals, whether any of these failure
modes are present in *your* run — and what to do about them.

Everything is plain statistics: **no model calls, no GPU, no external APIs, no datasets**.
The core is numpy + scipy; figures use matplotlib.

## What it audits

| Audit | Statistic | Output |
|---|---|---|
| **Position / label bias** | χ² goodness-of-fit of selection rates vs uniform, Cohen's w, per-position excess with cluster-bootstrap CIs | verdict + which slot is over-picked |
| **Answer-key balance** (dataset side) | χ² GOF of gold positions vs uniform | flags imbalanced benchmarks |
| **Length bias** | selection rate by option-length rank (χ²) + mean length z-score of the picked option (within-question paired t-test) | verdict + longest/shortest preference |
| **Gold-length artifact** (dataset side) | χ² GOF of the gold answer's length rank | flags "longest answer is correct" benchmarks |
| **Ordering consistency** | same-answer-content rate across shuffled orderings of the same question, per-question chance level, McNemar test on paired correctness | detects order-sensitive models and canonical-order memorization |
| **Confidence calibration** | ECE / MCE (equal-width or equal-mass bins), reliability diagram, bootstrap CI | over- vs under-confidence |
| **Selective prediction** | risk–coverage curve, AURC / E-AURC, maximum-coverage confidence threshold for a target risk | operational abstention rule |

All confidence intervals are **cluster bootstrap** percentile intervals with questions as
clusters, so re-ordered variants of one question — which are strongly correlated — move
together. Hypothesis tests use one ordering per question to keep rows independent (testing
on all rows of a multi-variant run would overstate significance).

## Install

```bash
pip install -e .          # numpy, scipy, matplotlib
```

Requires Python ≥ 3.9.

## Quickstart

**1. Generate two synthetic demo runs and their audit reports** (a clean model and a model
with five injected pathologies), written to `examples/`:

```bash
python -m choice_eval demo --out examples --n 1200 --variants 4
```

**2. Audit your own evaluation log:**

```bash
python -m choice_eval audit my_run.jsonl --out my_report --binning equal_mass
```

Each run produces a `report.md` with figures:

| Section | Clean demo run | Biased demo run |
|---|---|---|
| Position bias | none (p = 0.95) | **moderate** — option C over-picked, excess +0.094 [0.077, 0.112] |
| Length bias | none (p = 0.58) | **moderate** — longest option 0.312 vs uniform 0.250 |
| Ordering consistency | 0.909, mostly stable | **0.551, unstable**; McNemar p = 1.4e-06 (canonical-order advantage) |
| Calibration | ECE 0.039, slightly miscalibrated | **ECE 0.190, overconfident / poorly calibrated** |
| Selective prediction | threshold 0.37 → 58% coverage @ 15% risk | no useful operating point — recalibrate first |

Side-by-side reports live in [`examples/clean/`](examples/clean/report.md) and
[`examples/biased/`](examples/biased/report.md).

![Position bias in the biased demo run](examples/biased/fig_position.png)

## Input schema

JSONL, one response per line — one (question, option-ordering) observation:

```json
{"question_id": "q00042", "n_options": 4, "gold_index": 2, "selected_index": 0,
 "confidence": 0.83, "option_lengths": [41, 87, 63, 30],
 "option_ids": ["q00042_opt1", "q00042_opt0", "q00042_opt3", "q00042_opt2"],
 "variant_id": "shuffle_2"}
```

| Field | Required | Meaning |
|---|---|---|
| `question_id` | ✓ | groups re-ordered variants of the same question |
| `n_options` | ✓ | number of options (must be constant within a run) |
| `gold_index` | ✓ | position of the correct option **in this presentation order** |
| `selected_index` | ✓ | position chosen; `null` = abstain / unparsable |
| `confidence` |  | stated probability for the selected answer, in [0, 1] |
| `option_lengths` |  | character length of each presented option (enables length audit) |
| `option_ids` |  | content identity of each presented option (enables the ordering audit) |
| `variant_id` |  | which reordering this is (e.g. `canonical`, `shuffle_1`, ...) |

Logging tips: record `option_ids` so answers can be matched by *content* across orderings;
ask each question under 2+ shuffled orderings if you want the ordering audit; include
`option_lengths` to expose length artifacts.

## Verified against known injected biases

Because the toolkit ships a generator that injects biases with **known magnitude**, its
correctness is checkable without any external ground truth: inject a bias, confirm the
audit recovers it. `tests/` does exactly this (33 tests), and `demo/recovery_table.py`
reproduces this table:

| Injected bias | Generator knob | Designed effect | Recovered (95% CI) | Verdict |
|---|---|---|---|---|
| Position pull toward option C | `position_attract={2: 0.30}` | +0.077 excess at C | +0.082 [+0.067, +0.096] | p = 2e-28 → moderate |
| Longest-option pull | `length_attract=0.25` | +0.063 excess at longest rank | +0.055 [+0.040, +0.070] | p = 6e-15 → moderate |
| Overconfident answers | `confidence_shift=0.18` | large positive ECE, `overconfident` | ECE 0.132 [0.122, 0.141] | overconfident → moderately miscalibrated |
| Canonical-order memorization | `canonical_bonus=0.12` | -0.066 accuracy drop on shuffles | -0.069 measured drop | McNemar p = 7e-61 → systematic |

Run the suite with `pytest` (a few seconds; everything is synthetic).

## Using it as a library

```python
from choice_eval import load_jsonl, run_audit, write_report

run = load_jsonl("my_run.jsonl")           # or build EvalRun in code
bundle = run_audit(run, n_boot=1000)       # all seven audits at once
bundle.position.verdict                    # "none" | "minor" | "moderate" | "severe"
bundle.position.excess                     # per-position selection excess
bundle.calibration.ece_ci                  # bootstrap CI for ECE
bundle.abstention.suggested_threshold      # confidence cutoff for a target risk
write_report(bundle, "my_report/")         # report.md + figures
```

Individual audits (`audit_position`, `audit_length`, `audit_order`, `audit_calibration`,
`audit_abstention`) and the synthetic generator (`generate_run`) are importable too.

## Metric notes

- **ECE** — Σ over confidence bins of (bin share) × |bin accuracy − bin mean confidence|.
  Equal-width bins are the default; `--binning equal_mass` gives quantile bins when
  confidence piles up at 0.99.
- **Cohen's w** — √(χ²/N) effect size for the χ² tests; verdicts combine p < 0.01 with
  w thresholds (0.05 / 0.10 / 0.21 → none / minor / moderate / severe).
- **AURC / E-AURC** — area under the risk–coverage curve when answers are sorted by stated
  confidence; E-AURC is the excess over the oracle ordering (all correct answers first).
- **McNemar** — exact binomial test on the discordant correctness pairs of two orderings;
  a significant result means the model is systematically *better* under one ordering
  (often the dataset's canonical order — a memorization signature).
- **Chance consistency** — the ordering audit reports the per-question probability that two
  random picks coincide, so a 0.6 consistency on a 4-option question isn't mistaken for signal.

## Limitations

- One option count per run (4-way MCQ is the target case; mixed-K logs are rejected with a
  clear error).
- The ordering audit needs `option_ids`; the length audit needs `option_lengths`. Without
  them those sections are skipped with an explanation rather than guessed.
- χ² tests assume the one-ordering-per-question sampling; rates and CIs use all orderings.
- Verdict thresholds (w cutoffs, α = 0.01) are defaults, not laws — the point estimates and
  CIs are in the report for your own judgement.

## Project layout

```
choice_eval/
  schema.py        data model + JSONL I/O
  generators.py    synthetic runs with injectable, known biases
  position.py      position/label bias + answer-key balance
  length.py        length bias + gold-length artifact
  order.py         ordering consistency + McNemar
  calibration.py   ECE / MCE / reliability diagram
  abstention.py    risk–coverage, AURC, threshold optimization
  report.py        Markdown report + figures
  cli.py           `choice-eval audit` / `choice-eval demo`
tests/             33 tests incl. injection-recovery checks
demo/              recovery_table.py — regenerates the README table
examples/          demo output: two reports + figures
```

## References

- Zheng, L. et al. *Large Language Models Are Not Robust Multiple Choice Selectors*. ICLR 2024.
- Guo, C. et al. *On Calibration of Modern Neural Networks*. ICML 2017.
- Geifman, Y. & El-Yaniv, R. *Selective Prediction: A New View of Classification (and Regression)*. 2017.
- Peyrard, M. et al. *Investigating the Simplest Case of Position Bias*. 2021.

## License

MIT — see [LICENSE](LICENSE).
