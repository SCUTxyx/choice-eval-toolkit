"""Calibration and selective-prediction audits."""

import numpy as np
import pytest

from choice_eval.abstention import audit_abstention
from choice_eval.calibration import audit_calibration
from choice_eval.generators import generate_run
from choice_eval.schema import EvalRun, Response


# ------------------------------------------------------------- calibration --

def test_clean_run_is_nearly_calibrated():
    run = generate_run(n_questions=6000, seed=21)
    audit = audit_calibration(run, n_boot=200, seed=0)
    assert audit.ece < 0.08
    assert audit.ece_ci[0] <= audit.ece <= audit.ece_ci[1]
    # branch means equal the design accuracy; allow sampling noise
    assert abs(audit.mean_confidence - audit.accuracy) < 0.03
    assert np.isnan(audit.bin_accuracy).sum() < audit.bin_counts.size  # most bins populated


def test_overconfidence_is_detected_and_quantified():
    run = generate_run(n_questions=3000, seed=21, confidence_shift=0.18)
    audit = audit_calibration(run, n_boot=200, seed=0)
    assert audit.ece > 0.10
    assert audit.direction == "overconfident"
    assert audit.verdict in ("moderately miscalibrated", "poorly calibrated")


def test_underconfidence_direction():
    run = generate_run(n_questions=3000, seed=21, confidence_shift=-0.15)
    audit = audit_calibration(run, n_boot=200, seed=0)
    assert audit.direction == "underconfident"
    assert audit.mean_confidence < audit.accuracy


def test_equal_mass_binning_has_balanced_counts():
    run = generate_run(n_questions=2000, seed=21)
    audit = audit_calibration(run, binning="equal_mass", n_boot=100, seed=0)
    counts = audit.bin_counts[audit.bin_counts > 0]
    assert counts.max() / counts.min() < 2.0


def test_calibration_requires_confidence():
    run = EvalRun(name="empty")
    run.add(Response("q1", 4, 0, 1))  # no confidence
    with pytest.raises(ValueError, match="confidence"):
        audit_calibration(run)


# -------------------------------------------------------------- abstention --

def _manual_run(confs, corrects):
    run = EvalRun(name="manual")
    for i, (c, ok) in enumerate(zip(confs, corrects)):
        run.add(
            Response(
                question_id=f"q{i}",
                n_options=4,
                gold_index=0,
                selected_index=None if ok is None else (0 if ok else 1),
                confidence=c,
            )
        )
    return run


def test_abstention_threshold_recovered_exactly():
    # 10 answers, the 4 most confident are correct, everything after is wrong.
    confs = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    corrects = [True, True, True, True, False, False, False, False, False, False]
    audit = audit_abstention(_manual_run(confs, corrects), target_risk=0.05)
    assert audit.suggested_threshold == pytest.approx(0.65)
    assert audit.coverage_at_target == pytest.approx(0.4)
    assert audit.risk_at_threshold == pytest.approx(0.0)


def test_abstention_infeasible_target_returns_nan():
    confs = [0.9, 0.1]
    corrects = [False, False]
    audit = audit_abstention(_manual_run(confs, corrects), target_risk=0.05)
    assert np.isnan(audit.suggested_threshold)


def test_abstention_on_generated_run():
    run = generate_run(n_questions=2000, seed=23)
    audit = audit_abstention(run, target_risk=0.15)
    full_risk = 1 - audit.accuracy
    assert 0.0 <= audit.aurc <= 1.0
    assert audit.e_aurc >= 0.0  # confidence must beat random ordering
    # risk at 30% coverage should be clearly below the answer-everything error rate
    low_cov_risk = float(np.interp(0.3, audit.coverage_curve, audit.risk_curve))
    assert low_cov_risk < full_risk * 0.7
    assert not np.isnan(audit.suggested_threshold)
