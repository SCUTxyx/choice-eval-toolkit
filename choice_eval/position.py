"""Position / label bias audit.

In multiple-choice evaluation, "position bias" and "label bias" are the same
phenomenon observed from two sides: the model prefers answers that appear at a
particular slot (first, last, C, ...). Two complementary tests are run:

1. **Marginal test** — are presented positions selected uniformly? (χ² GOF
   against uniform, Cohen's w, per-position excess with cluster-bootstrap CIs.)
   Confound: a *balanced answer key* is assumed — with an imbalanced key, a
   position-blind but accurate model still selects non-uniformly (it selects
   the gold more often, wherever the key puts it). The dataset-side audit
   (§2 of the report) flags exactly this condition.
2. **Gold-offset test (confound-free)** — among *wrong* answers, the relative
   position of the selection, ``(selected - gold) mod K``, must be uniform over
   the K-1 non-zero offsets, *regardless of answer-key balance and accuracy*.
   This test cannot be fooled by key imbalance. It also covers the marginal
   test's blind spot: with an imbalanced key, an absolute slot attractor
   produces a concentrated offset mix and is caught here.

The two tests cover each other's confounds: run both, read them together.

Statistical design: hypothesis tests use ONE ordering per question (re-ordered
variants are correlated; testing on all rows would overstate significance) —
conservative on multi-variant runs. Rates, excesses and all CIs use every
answered response with question-level cluster bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .arrays import RunArrays, first_variant_mask, to_arrays
from .schema import EvalRun
from .stats import chi2_uniform, cluster_bootstrap_ci_vec, verdict_from_effect

_SEVERITY = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def _combine_verdict(a: str, b: str) -> str:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


@dataclass
class PositionAudit:
    k: int
    n_answered: int
    skipped: str | None
    # marginal (slot-level) test
    selection_rates: np.ndarray  # (k,) all answered responses
    selection_ci: np.ndarray  # (k, 2) cluster bootstrap
    chi2: float
    p_value: float
    effect_w: float
    marginal_verdict: str
    # gold-offset conditional test (confound-free)
    n_wrong: int
    offset_rates: np.ndarray  # (k-1,) P(selected = gold + d | wrong)
    offset_ci: np.ndarray  # (k-1, 2)
    offset_chi2: float
    offset_p: float
    offset_w: float
    offset_verdict: str
    # combined
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
    nan_off = np.full(max(k - 1, 1), np.nan)
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
        marginal_verdict = verdict_from_effect(p, w)

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

        # Gold-offset conditional test: among wrong answers, (sel - gold) mod K
        # must be uniform over the non-zero offsets, whatever the key looks like.
        wrong = test_rows & answered & ~arr.correct.astype(bool)
        n_wrong = int(wrong.sum())
        if n_wrong:
            offsets = (arr.sel[wrong] - arr.gold[wrong]) % k  # in 1..k-1
            offset_counts = np.bincount(offsets, minlength=k)[1:].astype(float)
            offset_rates = offset_counts / offset_counts.sum()
            offset_chi2, offset_p, offset_w = chi2_uniform(offset_counts)
            offset_verdict = verdict_from_effect(offset_p, offset_w)

            def _offset_rates(idx: np.ndarray) -> np.ndarray:
                o = offsets[idx]
                c = np.bincount(o, minlength=k)[1:]
                return c / c.sum()

            offset_ci = cluster_bootstrap_ci_vec(_offset_rates, arr.qid[wrong], n_boot, alpha, seed + 301)
        else:
            n_wrong = 0
            offset_rates = nan_off.copy()
            offset_ci = np.column_stack([nan_off, nan_off])
            offset_chi2 = offset_p = offset_w = float("nan")
            offset_verdict = "not available"  # nothing to condition on
    else:
        rates = nan_vec.copy()
        rates_ci = np.column_stack([nan_vec, nan_vec])
        chi2 = p = w = float("nan")
        marginal_verdict = "not available"
        excess = nan_vec.copy()
        excess_ci = np.column_stack([nan_vec, nan_vec])
        n_wrong = 0
        offset_rates = nan_off.copy()
        offset_ci = np.column_stack([nan_off, nan_off])
        offset_chi2 = offset_p = offset_w = float("nan")
        offset_verdict = "not available"

    if skipped is None:
        # Combined verdict — the model-side conclusion about position bias:
        # - the gold-offset test is confound-free: when it fires, bias is real;
        # - the marginal test is confound-free only under a balanced key;
        # - marginal signal + imbalanced key + silent offset test cannot be
        #   attributed (high accuracy on a skewed key looks identical), so the
        #   honest verdict is "inconclusive".
        key_imbalanced = gold_verdict in ("minor", "moderate", "severe")
        if offset_verdict in _SEVERITY and offset_verdict != "none":
            verdict = offset_verdict
        elif marginal_verdict in _SEVERITY and marginal_verdict != "none":
            verdict = marginal_verdict if not key_imbalanced else "inconclusive"
        else:
            verdict = "none"
    else:
        verdict = "not available"

    return PositionAudit(
        k=k,
        n_answered=n,
        skipped=skipped,
        selection_rates=rates,
        selection_ci=rates_ci,
        chi2=chi2,
        p_value=p,
        effect_w=w,
        marginal_verdict=marginal_verdict,
        n_wrong=n_wrong,
        offset_rates=offset_rates,
        offset_ci=offset_ci,
        offset_chi2=offset_chi2,
        offset_p=offset_p,
        offset_w=offset_w,
        offset_verdict=offset_verdict,
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
