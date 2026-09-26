"""Length bias audit.

Two separable questions:

1. *Benchmark artifact (dataset side)* — is the gold answer disproportionately
   the longest (or shortest) option? Many benchmarks carry a "longest answer is
   correct" artifact that models learn to exploit.
2. *Model side* — does the model prefer long (or short) options independent of
   correctness? Measured two ways: selection rate by length rank, and the mean
   length z-score of the selected option relative to its own question's options
   (a within-question paired statistic, tested with a one-sample t-test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .arrays import RunArrays, first_variant_mask, to_arrays
from .schema import EvalRun
from .stats import chi2_uniform, cluster_bootstrap_ci, verdict_from_effect


@dataclass
class LengthAudit:
    k: int
    n_used: int
    mean_z: float
    mean_z_ci: tuple[float, float]
    z_p_value: float
    selection_by_rank: np.ndarray  # rate at rank 0=shortest .. k-1=longest
    selection_by_rank_ci: np.ndarray
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
    usable = arr.answered & ~np.isnan(arr.lengths).any(axis=1)
    sel = arr.sel[usable]
    qid = arr.qid[usable]
    lengths = arr.lengths[usable]
    n = int(usable.sum())

    ranks = np.argsort(np.argsort(lengths, axis=1, kind="stable"), axis=1)  # 0 = shortest
    sel_rank = ranks[np.arange(n), sel]

    mean_lengths = lengths.mean(axis=1)
    std_lengths = lengths.std(axis=1)
    ok = std_lengths > 0
    z = (lengths[np.arange(n), sel] - mean_lengths) / np.where(ok, std_lengths, 1.0)
    z = z[ok]
    qid_ok = qid[ok]

    # Hypothesis tests use ONE ordering per question: multi-variant runs are
    # correlated within a question and would make the chi-square/t-test
    # anti-conservative. Rates and bootstrap CIs use all usable responses.
    test_mask = (first_variant_mask(arr) & usable)[usable]  # aligned to usable rows
    test_rank = sel_rank[test_mask]
    test_z = z[test_mask[ok]]
    if len(test_z) > 1 and np.std(test_z) > 0:
        _, z_p = stats.ttest_1samp(test_z, 0.0)
        z_p = float(z_p)
    else:
        z_p = 1.0
    z_ci = cluster_bootstrap_ci(lambda idx: float(np.mean(z[idx])), qid_ok, n_boot, alpha, seed + 1).as_tuple()

    counts = np.bincount(sel_rank, minlength=k).astype(float)
    rank_rates = counts / max(n, 1)
    test_counts = np.bincount(test_rank, minlength=k).astype(float)
    rank_chi2, rank_p, rank_w = chi2_uniform(test_counts)
    verdict = verdict_from_effect(rank_p, rank_w)

    def _rank_rate(sample_idx: np.ndarray, r: int) -> float:
        s = sel_rank[sample_idx]
        return float(np.mean(s == r)) if len(s) else 0.0

    rank_ci = np.array(
        [cluster_bootstrap_ci(lambda idx, r=r: _rank_rate(idx, r), qid, n_boot, alpha, seed + 11 + r).as_tuple()
         for r in range(k)]
    )

    # Dataset side: is the gold answer at an extreme length rank?
    fv = first_variant_mask(arr)
    gold_ok = fv & ~np.isnan(arr.lengths).any(axis=1)
    gold_lengths = arr.lengths[gold_ok]
    gold_ranks = np.argsort(np.argsort(gold_lengths, axis=1, kind="stable"), axis=1)
    gold_sel_rank = gold_ranks[np.arange(gold_ok.sum()), arr.gold[gold_ok]]
    gold_counts = np.bincount(gold_sel_rank, minlength=k).astype(float)
    _, gold_p, _ = chi2_uniform(gold_counts)
    gold_artifact = "artifact" if gold_p < 0.01 else "balanced"

    return LengthAudit(
        k=k,
        n_used=n,
        mean_z=float(np.mean(z)),
        mean_z_ci=z_ci,
        z_p_value=z_p,
        selection_by_rank=rank_rates,
        selection_by_rank_ci=rank_ci,
        rank_chi2=rank_chi2,
        rank_p=rank_p,
        rank_w=rank_w,
        verdict=verdict,
        longest_rate=float(rank_rates[-1]),
        shortest_rate=float(rank_rates[0]),
        gold_rank_distribution=gold_counts / gold_counts.sum(),
        gold_rank_p=gold_p,
        gold_artifact=gold_artifact,
    )
