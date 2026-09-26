"""Position / label bias audit.

In multiple-choice evaluation, "position bias" and "label bias" are the same
phenomenon observed from two sides: the model prefers answers that appear at a
particular slot (first, last, C, ...). This audit measures three things:

1. *Model side* — are presented positions selected uniformly? (chi-square GOF
   against uniform, Cohen's w effect size, per-position bootstrap CIs.)
2. *Dataset side* — is the answer key balanced across positions? (An imbalanced
   key is a benchmark artifact independent of the model.)
3. *Accuracy interaction* — does accuracy depend on where the gold answer sits?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .arrays import RunArrays, first_variant_mask, to_arrays
from .schema import EvalRun
from .stats import chi2_uniform, cluster_bootstrap_ci, verdict_from_effect


@dataclass
class PositionAudit:
    k: int
    n_answered: int
    selection_rates: np.ndarray  # (k,)
    selection_ci: np.ndarray  # (k, 2)
    chi2: float
    p_value: float
    effect_w: float
    verdict: str
    excess: np.ndarray  # rate_c - mean(rate at other positions)
    excess_ci: np.ndarray  # (k, 2)
    first_rate: float
    gold_distribution: np.ndarray
    gold_chi2: float
    gold_p: float
    gold_w: float
    gold_verdict: str
    accuracy_by_position: np.ndarray  # (k,) accuracy when gold sits at this position
    accuracy_p: float


def audit_position(
    run: EvalRun,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PositionAudit:
    arr: RunArrays = to_arrays(run)
    k = arr.k
    answered = arr.answered
    sel = arr.sel[answered]
    qid = arr.qid[answered]
    n = int(answered.sum())

    # Chi-square tests need independent rows: multi-variant runs correlate
    # heavily within a question, so the tests use ONE ordering per question.
    # Rates, excesses and bootstrap CIs below use every answered response.
    test_rows = first_variant_mask(arr) & answered
    test_counts = np.bincount(arr.sel[test_rows], minlength=k).astype(float)
    counts_all = np.bincount(sel, minlength=k).astype(float)

    rates = counts_all / max(n, 1)
    chi2, p, w = chi2_uniform(test_counts)
    verdict = verdict_from_effect(p, w)

    def _rate_c(sample_idx: np.ndarray, c: int) -> float:
        s = sel[sample_idx]
        return float(np.mean(s == c)) if len(s) else 0.0

    def _excess(sample_idx: np.ndarray, c: int) -> float:
        s = sel[sample_idx]
        if not len(s):
            return 0.0
        r = np.array([np.mean(s == d) for d in range(k)])
        return float(r[c] - np.mean(np.delete(r, c)))

    sel_ci = np.array(
        [cluster_bootstrap_ci(lambda idx, c=c: _rate_c(idx, c), qid, n_boot, alpha, seed + 101 + c).as_tuple()
         for c in range(k)]
    )
    excess = np.array([_excess(np.arange(n), c) for c in range(k)])
    excess_ci = np.array(
        [cluster_bootstrap_ci(lambda idx, c=c: _excess(idx, c), qid, n_boot, alpha, seed + 201 + c).as_tuple()
         for c in range(k)]
    )

    # Dataset side: one row per question (gold position is a question property).
    fv = first_variant_mask(arr)
    gold_counts = np.bincount(arr.gold[fv], minlength=k).astype(float)
    gold_chi2, gold_p, gold_w = chi2_uniform(gold_counts)
    gold_verdict = verdict_from_effect(gold_p, gold_w)

    # Accuracy as a function of where the gold answer sits (independent rows
    # only, as above).
    acc_by_pos = np.zeros(k)
    table = np.zeros((k, 2))
    for pos in range(k):
        m = test_rows & (arr.gold == pos)
        table[pos, 0] = arr.correct[m].sum()
        table[pos, 1] = (~arr.correct.astype(bool) & m).sum()
        acc_by_pos[pos] = arr.correct[m].mean() if m.sum() else np.nan
    _, accuracy_p, _, _ = stats.chi2_contingency(table)

    return PositionAudit(
        k=k,
        n_answered=n,
        selection_rates=rates,
        selection_ci=sel_ci,
        chi2=chi2,
        p_value=p,
        effect_w=w,
        verdict=verdict,
        excess=excess,
        excess_ci=excess_ci,
        first_rate=float(rates[0]),
        gold_distribution=gold_counts / gold_counts.sum(),
        gold_chi2=gold_chi2,
        gold_p=gold_p,
        gold_w=gold_w,
        gold_verdict=gold_verdict,
        accuracy_by_position=acc_by_pos,
        accuracy_p=float(accuracy_p),
    )
