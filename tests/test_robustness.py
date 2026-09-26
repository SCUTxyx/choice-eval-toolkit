"""Robustness: graceful skips, input validation, permutation-test validity,
AUROC, and machine-readable output."""

import json

import numpy as np
import pytest

from choice_eval import __version__
from choice_eval.abstention import audit_abstention
from choice_eval.arrays import to_arrays
from choice_eval.calibration import audit_calibration, confidence_auroc
from choice_eval.generators import generate_run
from choice_eval.length import audit_length
from choice_eval.order import audit_order
from choice_eval.position import audit_position
from choice_eval.report import bundle_to_dict, run_audit, write_report
from choice_eval.schema import EvalRun, Response, load_jsonl, save_jsonl


def _minimal_log(path: str, with_conf: bool, with_lengths: bool, with_ids: bool) -> str:
    run = EvalRun(name="minimal")
    for i in range(200):
        run.add(
            Response(
                question_id=f"q{i:04d}",
                n_options=4,
                gold_index=i % 4,
                selected_index=(i + 1) % 4,
                confidence=(0.5 + 0.1 * (i % 3)) if with_conf else None,
                option_lengths=[10 + j * 5 for j in range(4)] if with_lengths else None,
                option_ids=[f"q{i}_opt{j}" for j in range(4)] if with_ids else None,
            )
        )
    save_jsonl(run, path)
    return path


# ------------------------------------------------------------------- skips --

def test_run_without_confidence_skips_calibration_and_abstention(tmp_path):
    run = load_jsonl(_minimal_log(str(tmp_path / "a.jsonl"), with_conf=False, with_lengths=True, with_ids=True))
    bundle = run_audit(run, n_boot=100, n_perm=200)
    assert bundle.calibration.skipped and bundle.abstention.skipped
    assert bundle.position.skipped is None  # answers exist
    report = write_report(bundle, tmp_path / "out")
    text = report.read_text(encoding="utf-8")
    assert "skipped" in text
    assert (tmp_path / "out" / "results.json").exists()


def test_run_without_lengths_skips_length_audit(tmp_path):
    run = load_jsonl(_minimal_log(str(tmp_path / "b.jsonl"), with_conf=True, with_lengths=False, with_ids=True))
    bundle = run_audit(run, n_boot=100, n_perm=200)
    assert bundle.length.skipped and "option_lengths" in bundle.length.skipped
    assert bundle.calibration.skipped is None


def test_all_abstain_run_skips_selection_statistics():
    run = EvalRun(name="abstain")
    for i in range(50):
        run.add(Response(f"q{i}", 4, i % 4, None))
    pa = audit_position(run, n_boot=50)
    assert pa.skipped and "no answered" in pa.skipped
    assert pa.gold_verdict == "none"  # dataset-side audit still runs
    assert np.isnan(pa.selection_rates).all()
    la = audit_length(run, n_boot=50)
    assert la.skipped


def test_missing_option_ids_skips_order_audit():
    run = generate_run(n_questions=100, n_variants=2, seed=1)
    for r in run.responses:
        r.option_ids = None
    audit = audit_order(run)
    assert audit.skipped and "option_ids" in audit.skipped


# --------------------------------------------------------------- validation --

def test_duplicate_question_variant_pairs_rejected():
    run = EvalRun(name="dup")
    run.add(Response("q1", 4, 0, 1, confidence=0.5))
    run.add(Response("q1", 4, 0, 1, confidence=0.5))  # same ordering twice
    with pytest.raises(ValueError, match="duplicate"):
        to_arrays(run)
    # distinct variant ids are fine
    run2 = EvalRun(name="ok")
    run2.add(Response("q1", 4, 0, 1, confidence=0.5, variant_id="canonical"))
    run2.add(Response("q1", 4, 0, 1, confidence=0.5, variant_id="shuffle_1"))
    to_arrays(run2)


# ------------------------------------------------------ permutation test ----

def test_direction_permutation_symmetric_runs():
    """Under the null the permutation p-values should not be systematically small."""
    ps = []
    for seed in range(8):
        run = generate_run(n_questions=300, n_variants=3, seed=100 + seed)
        audit = audit_order(run, n_boot=50, n_perm=300, seed=0)
        ps.append(audit.direction_p)
    assert min(ps) > 0.005
    assert sum(p > 0.05 for p in ps) >= 7


def test_direction_permutation_detects_canonical_advantage():
    run = generate_run(n_questions=400, n_variants=4, seed=101, canonical_bonus=0.15)
    audit = audit_order(run, n_boot=100, n_perm=500, seed=0)
    assert audit.direction == "systematic direction"
    assert audit.direction_p < 0.01


def test_direction_permutation_v2_matches_exact_mcnemar():
    run = generate_run(n_questions=500, n_variants=2, seed=102, canonical_bonus=0.15)
    audit = audit_order(run, n_boot=50, n_perm=2000, seed=0)
    assert audit.mcnemar_p < 0.01
    assert audit.direction_p < 0.05  # permutation approximates the exact test


# ------------------------------------------------------------------- AUROC --

def test_auroc_perfect_and_reversed_separation():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    assert confidence_auroc(conf, np.array([1, 1, 1, 1, 0, 0, 0, 0])) == pytest.approx(1.0)
    assert confidence_auroc(conf, np.array([0, 0, 0, 0, 1, 1, 1, 1])) == pytest.approx(0.0)


def test_auroc_on_generated_run_beats_chance():
    run = generate_run(n_questions=2000, seed=5)
    arr = to_arrays(run)
    valid = arr.answered & ~np.isnan(arr.conf)
    auroc = confidence_auroc(arr.conf[valid], arr.correct[valid])
    assert auroc > 0.7  # confident answers are genuinely more often correct


def test_calibration_reports_auroc_with_ci():
    run = generate_run(n_questions=1500, seed=5)
    audit = audit_calibration(run, n_boot=200, seed=0)
    assert 0.5 < audit.auroc <= 1.0
    assert audit.auroc_ci[0] <= audit.auroc <= audit.auroc_ci[1]


# ---------------------------------------------------------- reproducibility --

def test_results_json_and_params_block(tmp_path):
    run = generate_run(n_questions=300, n_variants=2, seed=9)
    bundle = run_audit(run, n_boot=100, n_perm=100, seed=3)
    d = bundle_to_dict(bundle)
    text = json.dumps(d)  # must be valid JSON (NaN -> null)
    assert "NaN" not in text
    assert d["params"]["toolkit_version"] == __version__
    assert d["params"]["seed"] == 3
    assert d["position"]["verdict"] == bundle.position.verdict
    assert d["calibration"]["auroc"] == pytest.approx(bundle.calibration.auroc, rel=1e-6)

    report = write_report(bundle, tmp_path / "out")
    body = report.read_text(encoding="utf-8")
    assert f"v{__version__}" in body
    assert "results.json" in body
    assert (tmp_path / "out" / "results.json").exists()


def test_ordering_accuracy_table_populated_for_multivariant_runs():
    run = generate_run(n_questions=200, n_variants=3, seed=9)
    bundle = run_audit(run, n_boot=50, n_perm=50)
    assert len(bundle.ordering_accuracy) == 3
    names = [name for name, _, _ in bundle.ordering_accuracy]
    assert names[0] == "canonical"
    assert all(0.0 <= acc <= 1.0 for _, _, acc in bundle.ordering_accuracy)
