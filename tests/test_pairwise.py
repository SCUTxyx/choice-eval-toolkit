"""Pairwise (arena-style) audits against closed-form ground truth."""

import numpy as np
import pytest

from choice_eval.generators import (
    expected_pair_slot_rate,
    expected_p_chosen_longer,
    expected_swap_consistency,
    generate_pairwise,
)
from choice_eval.report import bundle_to_dict, run_pairwise_audit, write_pairwise_report
from choice_eval.schema import (
    PairJudgment,
    PairwiseRun,
    load_any,
    load_pairwise_jsonl,
    save_pairwise_jsonl,
)

N_BOOT = 200


# ------------------------------------------------------------- closed forms --

def test_clean_judge_passes_every_audit():
    run = generate_pairwise(n_pairs=3000, seed=21, discernment=0.75)
    bundle = run_pairwise_audit(run, n_boot=N_BOOT, seed=0)
    pa = bundle.pairwise
    assert pa.slot_verdict == "none"
    assert pa.slot_balance == "balanced"
    assert pa.swap_consistency == pytest.approx(expected_swap_consistency(0.75, 0.0), abs=0.02)
    assert pa.swap_verdict == "mostly stable"
    # pooled contents add length-realization noise: |rate - 0.5| stays small
    # and the content-clustered CI covers 0.5
    assert abs(pa.p_chosen_longer - 0.5) < 0.05
    assert pa.p_chosen_longer_ci[0] <= 0.5 <= pa.p_chosen_longer_ci[1] + 0.02
    assert pa.tie_rate == 0.0


def test_slot_pref_recovered_exactly():
    s = 0.20
    run = generate_pairwise(n_pairs=4000, seed=21, discernment=0.75, slot_pref=s)
    bundle = run_pairwise_audit(run, n_boot=N_BOOT, seed=0)
    pa = bundle.pairwise
    assert pa.pick_first_rate == pytest.approx(expected_pair_slot_rate(s), abs=0.015)
    assert pa.pick_first_ci[0] > 0.5  # CI excludes the coin flip
    assert pa.slot_verdict in ("moderate", "severe")
    assert pa.slot_balance == "balanced"  # design side still balanced


def test_order_flip_recovered_through_swap_consistency():
    f = 0.40
    run = generate_pairwise(n_pairs=3000, seed=21, discernment=0.75, order_flip=f)
    bundle = run_pairwise_audit(run, n_boot=N_BOOT, seed=0)
    pa = bundle.pairwise
    assert pa.swap_consistency == pytest.approx(expected_swap_consistency(0.75, f), abs=0.02)
    assert pa.swap_verdict == "unstable"
    assert pa.slot_verdict == "none"  # flips are symmetric, not a slot preference


def test_length_pref_recovered_exactly():
    lam = 0.35
    # n_contents = n_pairs: no shared contents, so the binomial closed form
    # holds exactly (with a shared pool, length–quality realization noise adds
    # a bias the CI accounts for — see the report's method notes)
    run = generate_pairwise(n_pairs=4000, seed=21, discernment=0.75, length_pref=lam,
                            n_contents=4000)
    bundle = run_pairwise_audit(run, n_boot=N_BOOT, seed=0)
    pa = bundle.pairwise
    assert pa.p_chosen_longer == pytest.approx(expected_p_chosen_longer(0.75, lam), abs=0.02)
    assert pa.p_chosen_longer_ci[0] > 0.5
    assert pa.length_verdict == "severe"
    assert pa.mean_length_advantage > 0


def test_length_pref_with_shared_pool_not_overclaimed():
    """With a pooled design, length realization noise shifts the rate; the
    content-clustered CI must still cover the designed effect."""
    lam = 0.30
    run = generate_pairwise(n_pairs=4000, seed=21, discernment=0.75, length_pref=lam,
                            n_contents=100)
    bundle = run_pairwise_audit(run, n_boot=N_BOOT, seed=0)
    pa = bundle.pairwise
    assert pa.p_chosen_longer > 0.5
    assert pa.p_chosen_longer_ci[1] >= expected_p_chosen_longer(0.75, lam) - 0.02
    assert pa.length_verdict in ("moderate", "severe")


def test_tie_rate_recorded():
    run = generate_pairwise(n_pairs=2000, seed=21, tie_prob=0.15)
    bundle = run_pairwise_audit(run, n_boot=100, seed=0)
    assert bundle.pairwise.tie_rate == pytest.approx(0.15, abs=0.02)


# ----------------------------------------------------- design-side balance --

def test_slot_assignment_exactly_balanced_by_generator():
    run = generate_pairwise(n_pairs=2000, seed=5, swap_orders=2)
    first = {}
    for j in run.judgments:
        first.setdefault(j.pair_id, j.slot_of_a)
    assert np.mean(list(first.values())) == pytest.approx(0.5, abs=0.02)


def test_slot_pref_confounded_when_design_imbalanced():
    """Hand-built: content A always presented first AND judge prefers slot 0 —
    the audit must flag the design so the slot signal is not over-read."""
    run = PairwiseRun(name="imbalanced")
    rng = np.random.default_rng(3)
    for i in range(2000):
        slot = 0  # A always first: design flaw
        pick_a = rng.random() < 0.6  # real preference for A + slot pull mixed
        run.add(PairJudgment(
            pair_id=f"p{i}", content_a_id="A", content_b_id="B",
            slot_of_a=slot, selected_slot=0 if pick_a else 1,
        ))
    bundle = run_pairwise_audit(run, n_boot=100, seed=0)
    pa = bundle.pairwise
    assert pa.slot_balance == "imbalanced"
    assert pa.slot_balance_p < 1e-6


# --------------------------------------------------- schema / io / report --

def test_pairwise_jsonl_roundtrip(tmp_path):
    run = generate_pairwise(n_pairs=20, seed=1)
    path = tmp_path / "pw.jsonl"
    save_pairwise_jsonl(run, path)
    loaded = load_pairwise_jsonl(path)
    assert len(loaded) == len(run)
    for a, b in zip(run.judgments, loaded.judgments):
        assert (a.pair_id, a.slot_of_a, a.selected_slot) == (b.pair_id, b.slot_of_a, b.selected_slot)


def test_load_any_detects_pairwise(tmp_path):
    run = generate_pairwise(n_pairs=5, seed=1)
    path = tmp_path / "pw.jsonl"
    save_pairwise_jsonl(run, path)
    kind, loaded = load_any(path)
    assert kind == "pairwise"
    assert len(loaded) == len(run)


def test_pair_judgment_validation():
    with pytest.raises(ValueError, match="slot_of_a"):
        PairwiseRun().add(PairJudgment("p", "a", "b", slot_of_a=2, selected_slot=0))
    with pytest.raises(ValueError, match="content_a_id"):
        PairwiseRun().add(PairJudgment("p", "same", "same", slot_of_a=0, selected_slot=0))
    with pytest.raises(ValueError, match="selected_slot"):
        PairwiseRun().add(PairJudgment("p", "a", "b", slot_of_a=0, selected_slot=3))


def test_pairwise_report_and_json_render(tmp_path):
    run = generate_pairwise(n_pairs=300, seed=7, slot_pref=0.2, length_pref=0.3)
    bundle = run_pairwise_audit(run, n_boot=100, seed=0)
    report = write_pairwise_report(bundle, tmp_path / "out")
    body = report.read_text(encoding="utf-8")
    assert "Pairwise Preference Audit" in body
    assert "Recommendations" in body
    assert (tmp_path / "out" / "fig_pair_slot.png").exists()
    assert (tmp_path / "out" / "fig_pair_winrates.png").exists()
    d = bundle_to_dict(bundle)
    text = json.dumps(d) if (json := __import__("json")) else ""
    assert "NaN" not in text
    assert d["pairwise"]["slot_verdict"] == bundle.pairwise.slot_verdict
    assert d["pairwise"]["content_win_rates"]


def test_swap_consistency_skipped_for_single_order():
    run = generate_pairwise(n_pairs=100, seed=1, swap_orders=1)
    bundle = run_pairwise_audit(run, n_boot=50, seed=0)
    assert bundle.pairwise.n_swap_pairs == 0
    body = write_pairwise_report(bundle, tmp_path := __import__("pathlib").Path("/tmp/pw_skip")).read_text(encoding="utf-8")
    assert "judge each pair under both presentation orders" in body
