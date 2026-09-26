"""Shared statistical helpers: cluster bootstrap, chi-square tests, McNemar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class CI:
    lo: float
    hi: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.lo, self.hi)


def cluster_bootstrap_ci(
    statistic,
    cluster_ids: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI for ``statistic(sample_idx)``.

    Resampling is done at the *cluster* level (typically question id), so
    correlated observations — e.g. several orderings of the same question —
    move together, which is the correct unit of resampling for MCQ runs.
    """
    cluster_ids = np.asarray(cluster_ids)
    unique = np.unique(cluster_ids)
    idx_by_cluster = {c: np.where(cluster_ids == c)[0] for c in unique}
    n_clusters = len(unique)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        sample_idx = np.concatenate([idx_by_cluster[unique[c]] for c in picked])
        draws[b] = statistic(sample_idx)
    return CI(*np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def chi2_uniform(counts: np.ndarray) -> tuple[float, float, float]:
    """Goodness-of-fit test against a uniform distribution.

    Returns (chi2 statistic, p-value, Cohen's w effect size).
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return 0.0, 1.0, 0.0
    expected = np.full_like(counts, n / len(counts))
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=len(counts) - 1))
    w = float(np.sqrt(chi2 / n))
    return chi2, p, w


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    """Exact McNemar test on discordant pair counts (b, c).

    Returns (chi2 statistic with continuity correction, two-sided p-value).
    """
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return float(chi2), p


def verdict_from_effect(p_value: float, effect: float, alpha: float = 0.01) -> str:
    """Combine p-value and Cohen's w into a plain-language verdict."""
    if p_value >= alpha or effect < 0.05:
        return "none"
    if effect < 0.10:
        return "minor"
    if effect < 0.21:
        return "moderate"
    return "severe"
