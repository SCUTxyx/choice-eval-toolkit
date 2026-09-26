"""Generators: determinism, statistical sanity, injection mechanics."""

import numpy as np

from choice_eval.arrays import to_arrays
from choice_eval.generators import generate_run


def test_deterministic_given_seed():
    a = generate_run(n_questions=50, n_variants=2, seed=3)
    b = generate_run(n_questions=50, n_variants=2, seed=3)
    ra = [(r.question_id, r.selected_index, r.confidence) for r in a.responses]
    rb = [(r.question_id, r.selected_index, r.confidence) for r in b.responses]
    assert ra == rb


def test_accuracy_near_design_value():
    run = generate_run(n_questions=8000, seed=1)
    arr = to_arrays(run)
    acc = arr.correct[arr.answered].mean()
    # designed: 0.55 * 0.88 + 0.45 * 0.30 = 0.619
    assert abs(acc - 0.619) < 0.02


def test_variants_share_lengths_and_gold_content():
    run = generate_run(n_questions=30, n_variants=3, seed=5)
    by_q = {}
    for r in run.responses:
        by_q.setdefault(r.question_id, []).append(r)
    for rows in by_q.values():
        canonical = rows[0]
        gold_content = canonical.option_ids[canonical.gold_index]
        gold_lengths = sorted(canonical.option_lengths)
        for r in rows[1:]:
            # same content set, permuted
            assert sorted(r.option_ids) == sorted(canonical.option_ids)
            assert sorted(r.option_lengths) == gold_lengths
            assert r.option_ids[r.gold_index] == gold_content


def test_position_attract_concentrates_selection():
    run = generate_run(n_questions=6000, seed=2, position_attract={2: 0.30})
    arr = to_arrays(run)
    sel = arr.sel[arr.answered]
    rates = np.bincount(sel, minlength=4) / len(sel)
    assert rates[2] == rates.max()
    assert rates[2] - np.delete(rates, 2).mean() > 0.05


def test_length_attract_prefers_longest():
    run = generate_run(n_questions=6000, seed=2, length_attract=0.30)
    arr = to_arrays(run)
    usable = arr.answered & ~np.isnan(arr.lengths).any(axis=1)
    lengths = arr.lengths[usable]
    sel = arr.sel[usable]
    ranks = np.argsort(np.argsort(lengths, axis=1, kind="stable"), axis=1)
    sel_rank = ranks[np.arange(len(sel)), sel]
    rates = np.bincount(sel_rank, minlength=4) / len(sel)
    assert rates[-1] == rates.max()
    assert rates[-1] - rates[:-1].mean() > 0.05


def test_confidence_shift_moves_mean_confidence():
    base = generate_run(n_questions=4000, seed=4)
    shifted = generate_run(n_questions=4000, seed=4, confidence_shift=0.18)
    arr_b, arr_s = to_arrays(base), to_arrays(shifted)
    conf_b = arr_b.conf[arr_b.answered].mean()
    conf_s = arr_s.conf[arr_s.answered].mean()
    # nominal +0.18, compressed a little by the 0.99 clip on the known branch
    assert 0.10 < conf_s - conf_b < 0.22
