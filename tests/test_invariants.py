"""Invariant fuzz: hammer run_audit across randomized generator configurations
and assert the structural invariants every audit must satisfy. Catches whole
classes of regressions (nan leaks, CIs not covering estimates, broken JSON,
rendering crashes) that point tests miss."""

import json

import numpy as np
import pytest

from choice_eval import generate_run
from choice_eval.report import bundle_to_dict, run_audit, write_report


def _random_kwargs(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 6))
    kwargs = {
        "n_questions": int(rng.integers(60, 260)),
        "k": k,
        "n_variants": int(rng.integers(1, 4)),
        "p_know": float(rng.uniform(0.2, 0.9)),
        "p_guess_correct": float(rng.uniform(0.1, 0.5)),
        "misconception_stick": float(rng.uniform(0.0, 1.0)),
        "length_attract": float(rng.uniform(0.0, 0.4)),
        "variant_flip": float(rng.uniform(0.0, 0.4)),
        "canonical_bonus": float(rng.uniform(0.0, 0.2)),
        "confidence_shift": float(rng.uniform(-0.25, 0.25)),
        "gold_long_bias": float(rng.uniform(0.0, 0.3)),
        "abstain_prob": float(rng.choice([0.0, 0.0, 0.05])),
    }
    if rng.random() < 0.4:
        kwargs["position_attract"] = {int(rng.integers(0, k)): 0.25}
    if rng.random() < 0.4 and k > 2:
        kwargs["offset_attract"] = {int(rng.integers(1, k)): 0.25}
    return kwargs


@pytest.mark.parametrize("seed", range(8))
def test_structural_invariants_across_random_configs(tmp_path, seed):
    run = generate_run(seed=seed, name=f"fuzz{seed}", **_random_kwargs(seed))
    bundle = run_audit(run, n_boot=120, n_perm=100, seed=seed)

    k = bundle.k
    pa, la, ca, ab = bundle.position, bundle.length, bundle.calibration, bundle.abstention

    if pa.skipped is None:
        assert pa.selection_rates.sum() == pytest.approx(1.0, abs=1e-9)
        assert ((pa.selection_rates >= 0) & (pa.selection_rates <= 1)).all()
        # CI must contain the point estimate, column by column
        assert (pa.selection_ci[:, 0] <= pa.selection_rates + 1e-12).all()
        assert (pa.selection_ci[:, 1] >= pa.selection_rates - 1e-12).all()
        assert (pa.excess_ci[:, 0] <= pa.excess + 1e-12).all()
        assert (pa.excess_ci[:, 1] >= pa.excess - 1e-12).all()
        if pa.n_wrong > 0:
            assert pa.offset_rates.sum() == pytest.approx(1.0, abs=1e-9)
            assert (pa.offset_ci[:, 0] <= pa.offset_rates + 1e-12).all()
            assert (pa.offset_ci[:, 1] >= pa.offset_rates - 1e-12).all()

    if la.skipped is None:
        assert la.selection_by_rank.sum() == pytest.approx(1.0, abs=1e-9)
        assert (la.selection_by_rank_ci[:, 0] <= la.selection_by_rank + 1e-12).all()
        assert (la.selection_by_rank_ci[:, 1] >= la.selection_by_rank - 1e-12).all()

    if ca.skipped is None:
        assert 0.0 <= ca.accuracy <= 1.0
        assert 0.0 <= ca.mean_confidence <= 1.0
        assert ca.ece >= 0.0
        assert ca.ece_ci[0] <= ca.ece + 1e-9
        assert np.isnan(ca.auroc) or 0.0 <= ca.auroc <= 1.0

    if ab.skipped is None:
        assert 0.0 <= ab.coverage_at_target <= 1.0
        assert ab.e_aurc >= -1e-12
        if not np.isnan(ab.suggested_threshold):
            assert ab.risk_at_threshold <= ab.target_risk + 1e-9

    # bundle must survive a JSON round trip without NaN literals
    text = json.dumps(bundle_to_dict(bundle))
    assert "NaN" not in text and "Infinity" not in text

    # and the report must render for every configuration
    report = write_report(bundle, tmp_path / f"fuzz{seed}")
    body = report.read_text(encoding="utf-8")
    assert "Summary" in body and "Recommendations" in body


def test_wide_run_renders_with_generated_slot_labels(tmp_path):
    """k = 30: slot labels must not produce garbage characters."""
    run = generate_run(n_questions=60, k=30, seed=1)
    bundle = run_audit(run, n_boot=50, n_perm=50)
    body = write_report(bundle, tmp_path / "wide").read_text(encoding="utf-8")
    assert "L27" in body  # 27th slot uses the fallback label
    assert "L30" in body
