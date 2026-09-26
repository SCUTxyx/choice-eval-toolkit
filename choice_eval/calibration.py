"""Confidence calibration audit: ECE / MCE / reliability diagram / AUROC.

Two complementary properties of a confidence score are measured:

* *Calibration* — when the model says 80%, is it right 80% of the time?
  (ECE / MCE over confidence bins, reliability diagram data.)
* *Discrimination* — does higher confidence mean more likely correct at all?
  (AUROC of confidence for correctness; a model can be badly calibrated yet
  still rank its answers perfectly, or vice versa.)

Only answered questions with a stated confidence are used. CIs use a cluster
bootstrap over questions, matching the correlated structure of multi-variant
runs. ECE is a slightly optimistic-biased estimator at small sample sizes
(see method notes in the report); the bootstrap CI conveys sampling error but
not this bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .arrays import RunArrays, to_arrays
from .schema import EvalRun
from .stats import cluster_bootstrap_ci


@dataclass
class CalibrationAudit:
    n: int
    skipped: str | None
    accuracy: float
    mean_confidence: float
    ece: float
    ece_ci: tuple[float, float]
    mce: float
    auroc: float
    auroc_ci: tuple[float, float]
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


def confidence_auroc(conf, correct) -> float:
    """P(higher confidence for a random correct answer than a random wrong one)."""
    pos = conf[correct == 1]
    neg = conf[correct == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (len(pos) * len(neg)))


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
    n = int(valid.sum())
    if n == 0:
        skipped = (
            "no answered responses carry confidence; log the model's stated "
            "probability for the selected answer to enable this audit"
        )
        nan = float("nan")
        return CalibrationAudit(
            n=0, skipped=skipped, accuracy=nan, mean_confidence=nan,
            ece=nan, ece_ci=(nan, nan), mce=nan, auroc=nan, auroc_ci=(nan, nan),
            direction="not available", binning=binning,
            bin_edges=np.linspace(0, 1, n_bins + 1),
            bin_accuracy=np.full(n_bins, nan), bin_confidence=np.full(n_bins, nan),
            bin_counts=np.zeros(n_bins, dtype=int), verdict="not available",
        )

    conf = arr.conf[valid]
    correct = arr.correct[valid].astype(float)
    qid = arr.qid[valid]

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

    auroc = confidence_auroc(conf, correct)
    if np.isnan(auroc):
        auroc_ci = (float("nan"), float("nan"))
    else:
        auroc_ci = cluster_bootstrap_ci(
            lambda idx_: confidence_auroc(conf[idx_], correct[idx_]),
            qid, n_boot, alpha, seed + 3,
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
        skipped=None,
        accuracy=acc,
        mean_confidence=mean_conf,
        ece=ece,
        ece_ci=ece_ci,
        mce=mce,
        auroc=auroc,
        auroc_ci=auroc_ci,
        direction=direction,
        binning=binning,
        bin_edges=edges,
        bin_accuracy=bin_acc,
        bin_confidence=bin_conf,
        bin_counts=bin_counts,
        verdict=verdict,
    )
