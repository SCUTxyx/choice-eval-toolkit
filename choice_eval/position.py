"""Position / label bias audit.

In multiple-choice evaluation, "position bias" and "label bias" are the same
phenomenon observed from two sides: the model prefers answers that appear at a
particular slot (first, last, C, ...). This audit measures three things:

1. *Model side* — are presented positions selected uniformly? (chi-square GOF
   against uniform, Cohen's w effect size, per-position bootstrap CIs.)
2. *Dataset side* — is the answer key balanced across positions? (An imbalanced
   key is a benchmark artifact independent of the model.)
3. *Accuracy interaction* — does accuracy depend on where the gold answer sits?

Statistical design: the chi-square test uses ONE ordering per question —
re-ordered variants of the same question are strongly correlated, and testing
on all rows would overstate significance. Selection rates, excesses and all
confidence intervals use every answered response, with question-level cluster
bootstrap. The test is therefore conservative on multi-variant runs; the
per-position CIs provide the all-data view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .arrays import RunArrays, first_variant_mask, to_arrays
from .schema import EvalRun
from .stats import chi2_uniform, cluster_bootstrap_ci_vec, verdict_from_effect


@dataclass
class PositionAudit:
    k: int
    n_answered: int
    skipped: str | None
    selection_rates: np.ndarray  # (k,) all answered responses
    selection_ci: np.ndarray  # (k, 2) cluster bootstrap
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
    n = int(answered.sum())

    # Dataset side: one row per question (gold position is a question property).
    fv = first_variant_mask(arr)
    gold_counts = np.bincount(arr.gold[fv], minlength=k).astype(float)
    gold_chi2, gold_p, gold_w = chi2_uniform(gold_counts)
    gold_verdict = verdict_from_effect(gold_p, gold_w)

    nan_vec = np.full(k, np.nan)
    skipped: str | None = None
    acc_by_pos = nan_vec.copy()
    accuracy_p = float("nan")
    if n == 0:
        skipped = "no answered responses (all abstain/unparsable); selection statistics unavailable"

    if skipped is None:
        sel = arr.sel[answered]
        qid = arr.qid[answered]
        rates = np.bincount(sel, minlength=k) / n

        test_rows = fv & answered
        test_counts = np.bincount(arr.sel[test_rows], minlength=k).astype(float)
        chi2, p, w = chi2_uniform(test_counts)
        verdict = verdict_from_effect(p, w)

        def _rates(idx: np.ndarray) -> np.ndarray:
            s = sel[idx]
            return np.bincount(s, minlength=k) / len(s)

        def _excess(idx: np.ndarray) -> np.ndarray:
            r = _rates(idx)
            return r - (r.sum() - r) / (k - 1)

        rates_ci = cluster_bootstrap_ci_vec(_rates, qid, n_boot, alpha, seed + 101)
        excess = _excess(np.arange(n))
        excess_ci = cluster_bootstrap_ci_vec(_excess, qid, n_boot, alpha, seed + 201)

        acc_by_pos = np.full(k, np.nan)
        table = np.zeros((k, 2))
        for pos in range(k):
            m = test_rows & (arr.gold == pos)
            if m.sum():
                acc_by_pos[pos] = arr.correct[m].mean()
                table[pos, 0] = arr.correct[m].sum()
                table[pos, 1] = (~arr.correct.astype(bool) & m).sum()
        if (table.sum(axis=1) > 0).all() and (table.sum(axis=0) > 0).all():
            # degenerate tables (0% or 100% accuracy) carry no homogeneity signal
            _, accuracy_p, _, _ = stats.chi2_contingency(table)
            accuracy_p = float(accuracy_p)
        else:
            accuracy_p = float("nan")
    else:
        rates = nan_vec.copy()
        rates_ci = np.column_stack([nan_vec, nan_vec])
        chi2 = p = w = float("nan")
        verdict = "not available"
        excess = nan_vec.copy()
        excess_ci = np.column_stack([nan_vec, nan_vec])

    return PositionAudit(
        k=k,
        n_answered=n,
        skipped=skipped,
        selection_rates=rates,
        selection_ci=rates_ci,
        chi2=chi2,
        p_value=p,
        effect_w=w,
        verdict=verdict,
        excess=excess,
        excess_ci=excess_ci,
        first_rate=float(rates[0]) if n else float("nan"),
        gold_distribution=gold_counts / gold_counts.sum(),
        gold_chi2=gold_chi2,
        gold_p=gold_p,
        gold_w=gold_w,
        gold_verdict=gold_verdict,
        accuracy_by_position=acc_by_pos,
        accuracy_p=accuracy_p,
    )
