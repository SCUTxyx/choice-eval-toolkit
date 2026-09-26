"""Selective prediction / abstention audit.

Given per-question confidence, sort answers from most to least confident and
trace the risk-coverage curve: how does error rate grow as more questions are
accepted? Reports AURC/E-AURC and the confidence threshold that achieves a
target risk with maximum coverage — the operational output most teams want.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arrays import RunArrays, to_arrays
from .schema import EvalRun


@dataclass
class AbstentionAudit:
    n: int
    abstain_rate: float
    accuracy: float
    aurc: float
    e_aurc: float  # excess AURC over the oracle ordering (>= 0)
    target_risk: float
    suggested_threshold: float
    coverage_at_target: float
    risk_at_threshold: float
    coverage_curve: np.ndarray  # downsampled for plotting
    risk_curve: np.ndarray


def audit_abstention(run: EvalRun, target_risk: float = 0.15) -> AbstentionAudit:
    arr: RunArrays = to_arrays(run)
    valid = arr.answered & ~np.isnan(arr.conf)
    conf = arr.conf[valid]
    correct = arr.correct[valid].astype(bool)
    n = int(valid.sum())
    if n == 0:
        raise ValueError("no answered responses with confidence; selective prediction cannot be computed")

    order = np.argsort(-conf, kind="stable")
    correct_sorted = correct[order].astype(float)
    conf_sorted = conf[order]

    cum_correct = np.cumsum(correct_sorted)
    sizes = np.arange(1, n + 1)
    risk = 1.0 - cum_correct / sizes  # error rate among the top-i most confident answers
    coverage = sizes / n

    aurc = float(risk.mean())

    # Oracle ordering: all correct answers first. E-AURC is the excess AURC
    # over this oracle (>= 0); the gap to a *random* ordering is aurc - (1 - acc).
    n_correct = int(correct_sorted.sum())
    oracle_risk = np.zeros(n)
    idx = np.arange(1, n + 1)
    beyond = idx > n_correct
    oracle_risk[beyond] = (idx[beyond] - n_correct) / idx[beyond]
    aurc_oracle = float(oracle_risk.mean())
    e_aurc = aurc - aurc_oracle

    # Maximum-coverage prefix whose risk stays within target.
    feasible = np.flatnonzero(risk <= target_risk)
    if len(feasible):
        keep = int(feasible[-1]) + 1  # prefix length (1-based)
        suggested = float(conf_sorted[keep - 1])  # lowest confidence still kept
        cov_at_target = float(keep / n)
        risk_at = float(risk[keep - 1])
    else:
        suggested = float("nan")
        cov_at_target = 0.0
        risk_at = float(risk[0])

    step = max(1, n // 200)
    return AbstentionAudit(
        n=n,
        abstain_rate=float(np.mean(~arr.answered | np.isnan(arr.conf))),
        accuracy=float(correct.mean()),
        aurc=aurc,
        e_aurc=e_aurc,
        target_risk=target_risk,
        suggested_threshold=suggested,
        coverage_at_target=cov_at_target,
        risk_at_threshold=risk_at,
        coverage_curve=coverage[::step],
        risk_curve=risk[::step],
    )
