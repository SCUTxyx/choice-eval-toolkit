"""End-to-end recovery: inject known biases, check the full pipeline finds them.

This is the toolkit's acceptance test — the synthetic equivalent of grading the
tool against an answer key it cannot see while running.
"""

import numpy as np

from choice_eval.generators import expected_selection_rates, generate_run
from choice_eval.report import run_audit

N_BOOT = 200


def test_clean_run_passes_every_audit():
    bundle = run_audit(generate_run(n_questions=2000, n_variants=4, seed=31), n_boot=N_BOOT)
    assert bundle.position.p_value > 0.01 and bundle.position.verdict == "none"
    assert bundle.position.gold_verdict == "none"
    assert bundle.length.rank_p > 0.01 and bundle.length.verdict == "none"
    assert bundle.length.gold_artifact == "balanced"
    assert bundle.order.content_consistency > 0.90
    assert bundle.order.mcnemar_p > 0.01
    assert bundle.calibration.ece < 0.08


def test_biased_run_fails_every_audit_and_recover_magnitudes():
    kwargs = dict(
        position_attract={2: 0.30},
        length_attract=0.25,
        confidence_shift=0.18,
        variant_flip=0.50,
        canonical_bonus=0.12,
    )
    run = generate_run(n_questions=6000, n_variants=4, seed=31, **kwargs)
    bundle = run_audit(run, n_boot=N_BOOT)

    acc = bundle.accuracy

    # position: detected at the injected slot with the predicted magnitude
    assert bundle.position.p_value < 1e-6
    assert bundle.position.verdict in ("moderate", "severe")
    assert int(np.argmax(bundle.position.excess)) == 2
    expected_pos = expected_selection_rates(acc, 4, position_attract={2: 0.30})
    assert np.allclose(bundle.position.selection_rates, expected_pos, atol=0.02)

    # length: detected at the longest rank with the predicted magnitude
    assert bundle.length.rank_p < 1e-6
    expected_len = expected_selection_rates(
        acc, 4, position_attract={2: 0.30}, length_attract=0.25, space="rank"
    )
    assert np.allclose(bundle.length.selection_by_rank, expected_len, atol=0.02)

    # ordering: unstable, and the canonical-order advantage is directional
    assert bundle.order.content_consistency < 0.85
    assert bundle.order.mcnemar_p < 1e-4
    # calibration: confidently wrong
    assert bundle.calibration.ece > 0.10
    assert bundle.calibration.direction == "overconfident"

    # selective prediction: confidence is too miscalibrated for a 15% target —
    # every tie-consistent threshold cut exceeds it, so the honest output is
    # "infeasible" (the pre-0.3.0 prefix logic would have reported a cut that
    # no deployable "conf >= t" rule could reproduce)
    assert bundle.abstention.e_aurc >= 0.0
    assert np.isnan(bundle.abstention.suggested_threshold)


def test_report_renders_for_both_runs(tmp_path):
    from choice_eval.report import write_report

    for name, kwargs in [("clean", {}), ("biased", dict(position_attract={2: 0.30}, confidence_shift=0.18))]:
        run = generate_run(n_questions=400, n_variants=2, seed=41, **kwargs)
        bundle = run_audit(run, n_boot=100)
        path = write_report(bundle, tmp_path / name)
        text = path.read_text(encoding="utf-8")
        assert "Summary" in text and "Recommendations" in text
        for fig in ("fig_position.png", "fig_length.png", "fig_reliability.png", "fig_risk_coverage.png"):
            assert (tmp_path / name / fig).exists()
