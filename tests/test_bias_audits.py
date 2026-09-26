"""Bias audit functions on known ground truth.

The generator's wrong-answer priority is: position attractor -> longest wrong
option -> misconception -> uniform. The audits are graded against the exact
per-rank selection probabilities this implies, via
``generators.expected_selection_rates``.
"""

import numpy as np

from choice_eval.arrays import to_arrays
from choice_eval.generators import expected_selection_rates, generate_run
from choice_eval.position import audit_position
from choice_eval.length import audit_length
from choice_eval.order import audit_order


def _measured_accuracy(run):
    arr = to_arrays(run)
    return arr.correct[arr.answered].mean()


# ---------------------------------------------------------------- position --

def test_position_clean_run_is_not_flagged():
    run = generate_run(n_questions=3000, seed=11)
    audit = audit_position(run, n_boot=200, seed=0)
    assert audit.p_value > 0.01
    assert audit.verdict == "none"
    assert audit.gold_verdict == "none"  # answer key balanced


def test_position_injected_bias_is_recovered():
    b, k = 0.30, 4
    run = generate_run(n_questions=6000, seed=11, position_attract={2: b})
    audit = audit_position(run, n_boot=200, seed=0)

    assert audit.p_value < 1e-6
    assert audit.verdict in ("moderate", "severe")

    acc = _measured_accuracy(run)
    expected = expected_selection_rates(acc, k, position_attract={2: b})
    assert np.allclose(audit.selection_rates, expected, atol=0.015)
    lo, hi = audit.excess_ci[2]
    assert lo > 0
    assert int(np.argmax(audit.excess)) == 2


def test_answer_key_imbalance_flagged_on_dataset_side():
    skewed = generate_run(n_questions=3000, seed=11)
    for r in skewed.responses:
        if r.question_id.endswith(("0", "1", "2", "3", "4", "5", "6", "7")):
            r.gold_index = 0
    audit = audit_position(skewed, n_boot=200, seed=0)
    assert audit.gold_p < 1e-6
    assert audit.gold_verdict in ("moderate", "severe")


# ------------------------------------------------------------------ length --

def test_length_clean_run_is_not_flagged():
    run = generate_run(n_questions=3000, seed=13)
    audit = audit_length(run, n_boot=200, seed=0)
    assert audit.rank_p > 0.01
    assert audit.verdict == "none"
    assert audit.gold_artifact == "balanced"
    assert audit.mean_z_ci[0] < 0 < audit.mean_z_ci[1]


def test_length_injected_bias_is_recovered():
    l, k = 0.35, 4
    run = generate_run(n_questions=6000, seed=13, length_attract=l)
    audit = audit_length(run, n_boot=200, seed=0)

    assert audit.rank_p < 1e-6
    assert audit.verdict in ("moderate", "severe")

    acc = _measured_accuracy(run)
    expected = expected_selection_rates(acc, k, length_attract=l, space="rank")
    assert np.allclose(audit.selection_by_rank, expected, atol=0.015), (
        audit.selection_by_rank, expected
    )
    assert audit.mean_z > 0.1
    assert audit.mean_z_ci[0] > 0


def test_gold_length_artifact_detected():
    run = generate_run(n_questions=4000, seed=13, gold_long_bias=0.4)
    audit = audit_length(run, n_boot=200, seed=0)
    assert audit.gold_rank_p < 1e-6
    assert audit.gold_artifact == "artifact"


# ------------------------------------------------------------------- order --

def test_order_clean_run_is_stable_and_symmetric():
    run = generate_run(n_questions=800, n_variants=4, seed=17)
    audit = audit_order(run, n_boot=200, seed=0)
    assert audit.n_pairs == 800 * 6
    assert audit.content_consistency > 0.90
    assert audit.verdict in ("stable", "mostly stable")
    assert audit.mcnemar_p > 0.01  # no systematic direction


def test_order_detects_flips_and_canonical_advantage():
    run = generate_run(n_questions=800, n_variants=4, seed=17, variant_flip=0.50)
    audit = audit_order(run, n_boot=200, seed=0)
    assert audit.content_consistency < 0.85
    assert audit.verdict == "unstable"

    run2 = generate_run(n_questions=1600, n_variants=4, seed=17, canonical_bonus=0.15)
    audit2 = audit_order(run2, n_boot=200, seed=0)
    assert audit2.mcnemar_p < 1e-8


def test_order_skips_single_variant_runs():
    run = generate_run(n_questions=100, seed=17)
    audit = audit_order(run)
    assert audit.skipped is not None
    assert np.isnan(audit.content_consistency)
