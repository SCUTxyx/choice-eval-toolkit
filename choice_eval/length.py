"""Length bias audit.

Two separable questions:

1. *Benchmark artifact (dataset side)* — is the gold answer disproportionately
   the longest (or shortest) option? Many benchmarks carry a "longest answer is
   correct" artifact that models learn to exploit.
2. *Model side* — does the model prefer long (or short) options independent of
   correctness? Measured two ways: selection rate by length rank, and the mean
   length z-score of the selected option relative to its own question's options
   (a within-question paired statistic).

Statistical design mirrors the position audit: hypothesis tests use one
ordering per question (re-ordered variants are correlated); rates and CIs use
every usable response with question-level cluster bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .arrays import RunArrays, first_variant_mask, to_arrays
from .schema import EvalRun
from .stats import chi2_uniform, cluster_bootstrap_ci, cluster_bootstrap_ci_vec, verdict_from_effect


@dataclass
class LengthAudit:
    k: int
    n_used: int
    skipped: str | None
    mean_z: float
    mean_z_ci: tuple[float, float]
    z_p_value: float
    selection_by_rank: np.ndarray  # rate at rank 0=shortest .. k-1=longest
    selection_by_rank_ci: np.ndarray  # (k, 2)
    rank_chi2: float
    rank_p: float
    rank_w: float
    verdict: str
    longest_rate: float
    shortest_rate: float
    gold_rank_distribution: np.ndarray
    gold_rank_p: float
    gold_artifact: str


def audit_length(
    run: EvalRun,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> LengthAudit:
    arr: RunArrays = to_arrays(run)
    k = arr.k
    answered = arr.answered
    has_lengths = ~np.isnan(arr.lengths).any(axis=1)
    usable = answered & has_lengths
    n = int(usable.sum())

    # Dataset side: gold answer's length rank, one row per question.
    fv = first_variant_mask(arr)
    gold_usable = fv & has_lengths
    if gold_usable.sum():
        gold_lengths = arr.lengths[gold_usable]
        gold_ranks = np.argsort(np.argsort(gold_lengths, axis=1, kind="stable"), axis=1)
        gold_sel_rank = gold_ranks[np.arange(gold_usable.sum()), arr.gold[gold_usable]]
        gold_counts = np.bincount(gold_sel_rank, minlength=k).astype(float)
        _, gold_p, _ = chi2_uniform(gold_counts)
        gold_dist = gold_counts / gold_counts.sum()
    else:
        gold_p, gold_dist = float("nan"), np.full(k, np.nan)
    gold_artifact = "unknown" if np.isnan(gold_p) else ("artifact" if gold_p < 0.01 else "balanced")

    skipped: str | None = None
    if int(answered.sum()) == 0:
        skipped = "no answered responses; length statistics unavailable"
    elif n == 0:
        skipped = "records lack option_lengths; log the character length of every option to enable this audit"

    nan_vec = np.full(k, np.nan)
    mean_z = float("nan")
    z_ci = (float("nan"), float("nan"))
    z_p = float("nan")
    rank_rates = nan_vec.copy()
    rates_ci = np.column_stack([nan_vec, nan_vec])
    rank_chi2 = rank_p = rank_w = float("nan")
    verdict = "not available"

    if skipped is None:
        sel = arr.sel[usable]
        qid = arr.qid[usable]
        lengths = arr.lengths[usable]

        ranks = np.argsort(np.argsort(lengths, axis=1, kind="stable"), axis=1)  # 0 = shortest
        sel_rank = ranks[np.arange(n), sel]

        mean_lengths = lengths.mean(axis=1)
        std_lengths = lengths.std(axis=1)
        ok = std_lengths > 0
        z = (lengths[np.arange(n), sel] - mean_lengths) / np.where(ok, std_lengths, 1.0)

        # Hypothesis tests on one ordering per question (independent rows).
        test_mask = (first_variant_mask(arr) & usable)[usable]  # aligned to usable rows
        test_rank = sel_rank[test_mask]
        test_z = z[test_mask & ok]
        if len(test_z) > 1 and np.std(test_z) > 0:
            _, z_p = stats.ttest_1samp(test_z, 0.0)
            z_p = float(z_p)
        else:
            z_p = 1.0

        rank_rates = np.bincount(sel_rank, minlength=k).astype(float) / n
        test_counts = np.bincount(test_rank, minlength=k).astype(float)
        rank_chi2, rank_p, rank_w = chi2_uniform(test_counts)
        verdict = verdict_from_effect(rank_p, rank_w)

        def _rank_rates(idx: np.ndarray) -> np.ndarray:
            s = sel_rank[idx]
            return np.bincount(s, minlength=k) / len(s)

        rates_ci = cluster_bootstrap_ci_vec(_rank_rates, qid, n_boot, alpha, seed + 11)
        if ok.sum():
            z_ci = cluster_bootstrap_ci(
                lambda idx: float(np.mean(z[idx])), qid[ok], n_boot, alpha, seed + 1
            ).as_tuple()
        mean_z = float(np.mean(z)) if n else float("nan")

    return LengthAudit(
        k=k,
        n_used=n,
        skipped=skipped,
        mean_z=mean_z,
        mean_z_ci=z_ci,
        z_p_value=z_p,
        selection_by_rank=rank_rates,
        selection_by_rank_ci=rates_ci,
        rank_chi2=rank_chi2,
        rank_p=rank_p,
        rank_w=rank_w,
        verdict=verdict,
        longest_rate=float(rank_rates[-1]) if skipped is None else float("nan"),
        shortest_rate=float(rank_rates[0]) if skipped is None else float("nan"),
        gold_rank_distribution=gold_dist,
        gold_rank_p=gold_p,
        gold_artifact=gold_artifact,
    )
