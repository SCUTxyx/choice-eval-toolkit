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
    skipped: str | None
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
    n = int(valid.sum())
    abstain_rate = float(np.mean(~arr.answered | np.isnan(arr.conf)))
    nan = float("nan")
    if n == 0:
        skipped = (
            "no answered responses carry confidence; selective prediction "
            "cannot be computed without confidence values"
        )
        return AbstentionAudit(
            n=0, skipped=skipped, abstain_rate=abstain_rate, accuracy=nan,
            aurc=nan, e_aurc=nan, target_risk=target_risk,
            suggested_threshold=nan, coverage_at_target=nan, risk_at_threshold=nan,
            coverage_curve=np.array([0.0, 1.0]), risk_curve=np.array([nan, nan]),
        )

    conf = arr.conf[valid]
    correct = arr.correct[valid].astype(bool)

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

    # Suggested threshold: the *deployable rule* is "answer when conf >= t", so
    # the cut must fall between tied confidence values — otherwise the rule
    # selects more rows than the evaluated prefix and the stated risk is wrong.
    # Among tie-consistent cuts, pick the one with maximum coverage whose rule
    # risk stays within target.
    order_asc = np.argsort(conf, kind="stable")
    conf_asc = conf[order_asc]
    correct_asc = correct[order_asc].astype(float)
    cum_correct_asc = np.concatenate([[0.0], np.cumsum(correct_asc)])
    total_correct = float(correct_asc.sum())

    vals = np.unique(conf_asc)  # ascending
    starts = np.searchsorted(conf_asc, vals, side="left")  # first row with conf >= val
    sizes = n - starts
    correct_in_rule = total_correct - cum_correct_asc[starts]
    rule_risk = 1.0 - correct_in_rule / sizes
    feasible = rule_risk <= target_risk + 1e-12
    if feasible.any():
        j = int(np.flatnonzero(feasible)[0])  # smallest val = maximum coverage
        suggested = float(vals[j])
        cov_at_target = float(sizes[j] / n)
        risk_at = float(rule_risk[j])
    else:
        suggested = float("nan")
        cov_at_target = 0.0
        risk_at = float("nan")

    step = max(1, n // 200)
    return AbstentionAudit(
        n=n,
        skipped=None,
        abstain_rate=abstain_rate,
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
