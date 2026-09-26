"""Confidence calibration audit: ECE / MCE / reliability diagram data.

Calibration here answers: *when the model says 80%, is it right 80% of the
time?* Only answered questions with a stated confidence are used. CIs use a
cluster bootstrap over questions, matching the correlated structure of
multi-variant runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arrays import RunArrays, to_arrays
from .schema import EvalRun
from .stats import cluster_bootstrap_ci


@dataclass
class CalibrationAudit:
    n: int
    accuracy: float
    mean_confidence: float
    ece: float
    ece_ci: tuple[float, float]
    mce: float
    direction: str  # overconfident / underconfident / balanced
    binning: str
    bin_edges: np.ndarray
    bin_accuracy: np.ndarray  # nan = empty bin
    bin_confidence: np.ndarray
    bin_counts: np.ndarray
    verdict: str


def _bin_edges(conf: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    if binning == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if binning == "equal_mass":
        qs = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        return np.unique(qs)  # collapse duplicates from tied confidences
    raise ValueError(f"unknown binning {binning!r}")


def expected_calibration_error(conf, correct, edges) -> float:
    """ECE = sum over bins of (bin share) * |bin accuracy - bin mean confidence|."""
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    n = len(conf)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def audit_calibration(
    run: EvalRun,
    n_bins: int = 10,
    binning: str = "equal_width",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CalibrationAudit:
    arr: RunArrays = to_arrays(run)
    valid = arr.answered & ~np.isnan(arr.conf)
    conf = arr.conf[valid]
    correct = arr.correct[valid].astype(float)
    qid = arr.qid[valid]
    n = int(valid.sum())
    if n == 0:
        raise ValueError("no answered responses with confidence; calibration cannot be computed")

    edges = _bin_edges(conf, n_bins, binning)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    n_bins_eff = len(edges) - 1

    bin_acc = np.full(n_bins_eff, np.nan)
    bin_conf = np.full(n_bins_eff, np.nan)
    bin_counts = np.zeros(n_bins_eff, dtype=int)
    for b in range(n_bins_eff):
        m = idx == b
        bin_counts[b] = int(m.sum())
        if m.any():
            bin_acc[b] = correct[m].mean()
            bin_conf[b] = conf[m].mean()

    gaps = np.abs(bin_acc - bin_conf)
    ece = float(np.nansum((bin_counts / n) * gaps))
    mce = float(np.nanmax(gaps)) if np.any(bin_counts > 0) else 0.0

    ece_ci = cluster_bootstrap_ci(
        lambda idx_: expected_calibration_error(conf[idx_], correct[idx_], edges),
        qid, n_boot, alpha, seed,
    ).as_tuple()

    mean_conf = float(conf.mean())
    acc = float(correct.mean())
    diff = mean_conf - acc
    direction = "overconfident" if diff > 0.01 else ("underconfident" if diff < -0.01 else "balanced")

    if ece < 0.03:
        verdict = "well calibrated"
    elif ece < 0.08:
        verdict = "slightly miscalibrated"
    elif ece < 0.15:
        verdict = "moderately miscalibrated"
    else:
        verdict = "poorly calibrated"

    return CalibrationAudit(
        n=n,
        accuracy=acc,
        mean_confidence=mean_conf,
        ece=ece,
        ece_ci=ece_ci,
        mce=mce,
        direction=direction,
        binning=binning,
        bin_edges=edges,
        bin_accuracy=bin_acc,
        bin_confidence=bin_conf,
        bin_counts=bin_counts,
        verdict=verdict,
    )
