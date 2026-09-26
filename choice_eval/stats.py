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


def _cluster_groups(cluster_ids: np.ndarray) -> list[np.ndarray]:
    """Row indices per cluster, via one sort — O(N log N), not O(N x clusters)."""
    ids = np.asarray(cluster_ids)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    return np.split(order, boundaries)


def cluster_bootstrap_draws(stat_fn, cluster_ids: np.ndarray, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of ``stat_fn`` under question-level resampling.

    Resampling happens at the *cluster* level (typically question id), so
    correlated observations — e.g. several orderings of the same question —
    move together, which is the correct unit of resampling for MCQ runs.

    ``stat_fn(row_indices)`` may return a scalar or a 1-D vector (constant
    length across draws); the result has shape ``(B,)`` or ``(B, m)``
    respectively.
    """
    groups = _cluster_groups(cluster_ids)
    n_clusters = len(groups)
    rng = np.random.default_rng(seed)
    draws = []
    # Draw in chunks so the (chunk x clusters) index matrix stays small even
    # for runs with hundreds of thousands of questions.
    chunk = max(1, min(n_boot, max(1, 100_000 // max(n_clusters, 1))))
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        picks = rng.integers(0, n_clusters, size=(b, n_clusters))
        for row in picks:
            idx = np.concatenate([groups[c] for c in row])
            draws.append(np.atleast_1d(stat_fn(idx)))
        done += b
    # (B,) for scalar statistics or (B, m) for vector statistics — callers
    # disambiguate; no automatic flattening (a length-1 vector is not a scalar).
    return np.asarray(draws)


def cluster_bootstrap_ci(stat_fn, cluster_ids: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> CI:
    """Percentile bootstrap CI for a scalar statistic (see above for the design)."""
    draws = cluster_bootstrap_draws(stat_fn, cluster_ids, n_boot, seed)
    if draws.ndim != 2 or draws.shape[1] != 1:
        raise ValueError("cluster_bootstrap_ci expects a scalar statistic; "
                         "use cluster_bootstrap_ci_vec for vector statistics")
    lo, hi = np.percentile(draws[:, 0], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(float(lo), float(hi))


def cluster_bootstrap_ci_vec(stat_fn, cluster_ids: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> np.ndarray:
    """Percentile bootstrap CIs for a vector statistic; returns shape (m, 2)."""
    draws = cluster_bootstrap_draws(stat_fn, cluster_ids, n_boot, seed)
    if draws.ndim != 2:
        raise ValueError("cluster_bootstrap_ci_vec expects a vector statistic")
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return np.column_stack([lo, hi])


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
    Valid when each question contributes one independent pair; for runs with
    more than two orderings use the variant-permutation test in
    :mod:`choice_eval.order` instead.
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
