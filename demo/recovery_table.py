"""Print the injected-vs-recovered table used in the README.

Each row injects ONE bias into an otherwise clean synthetic run (n = 6000
questions) and reports what the audit measured. Excess CIs come from the
audit's cluster bootstrap (questions as clusters). Run:

    PYTHONPATH=. python demo/recovery_table.py
"""

from __future__ import annotations

import numpy as np

from choice_eval.arrays import to_arrays
from choice_eval.generators import expected_selection_rates, generate_run
from choice_eval.report import run_audit

N = 6000
SEED = 3
N_BOOT = 500


def _excess_ci_boot(values_by_row_sel, qid, slot, k, n_boot=N_BOOT, seed=1):
    """Cluster bootstrap CI for excess of `slot` over the mean of the others."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(qid, return_inverse=True)
    by_cluster = {i: np.where(inverse == i)[0] for i in range(len(unique))}
    draws = np.empty(n_boot)
    n = len(values_by_row_sel)
    for bidx in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([by_cluster[i] for i in picked])
        r = np.bincount(values_by_row_sel[idx], minlength=k) / len(idx)
        draws[bidx] = r[slot] - np.delete(r, slot).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _rate_excess(rates, slot):
    return rates[slot] - np.delete(rates, slot).mean()


def main() -> None:
    rows = []

    # 1. Position attractor at option C.
    run = generate_run(n_questions=N, seed=SEED, position_attract={2: 0.30})
    b = run_audit(run, n_boot=N_BOOT)
    exp = expected_selection_rates(b.accuracy, 4, position_attract={2: 0.30})
    arr = to_arrays(run)
    lo, hi = _excess_ci_boot(arr.sel[arr.answered], arr.qid[arr.answered], 2, 4)
    rows.append(
        (
            "Position pull toward option C",
            "`position_attract={2: 0.30}`",
            f"{_rate_excess(exp, 2):+.3f} excess at C",
            f"{_rate_excess(b.position.selection_rates, 2):+.3f} [{lo:+.3f}, {hi:+.3f}]",
            f"p = {b.position.p_value:.0e} → {b.position.verdict}",
        )
    )

    # 2. Longest-option attractor.
    run = generate_run(n_questions=N, seed=SEED, length_attract=0.25)
    b = run_audit(run, n_boot=N_BOOT)
    exp = expected_selection_rates(b.accuracy, 4, length_attract=0.25, space="rank")
    arr = to_arrays(run)
    usable = arr.answered & ~np.isnan(arr.lengths).any(axis=1)
    lengths = arr.lengths[usable]
    sel = arr.sel[usable]
    ranks = np.argsort(np.argsort(lengths, axis=1, kind="stable"), axis=1)
    sel_rank = ranks[np.arange(len(sel)), sel]
    lo, hi = _excess_ci_boot(sel_rank, arr.qid[usable], 3, 4)
    rows.append(
        (
            "Longest-option pull",
            "`length_attract=0.25`",
            f"{_rate_excess(exp, 3):+.3f} excess at longest rank",
            f"{_rate_excess(b.length.selection_by_rank, 3):+.3f} [{lo:+.3f}, {hi:+.3f}]",
            f"p = {b.length.rank_p:.0e} → {b.length.verdict}",
        )
    )

    # 3. Overconfidence.
    run = generate_run(n_questions=N, seed=SEED, confidence_shift=0.18)
    b = run_audit(run, n_boot=N_BOOT)
    lo, hi = b.calibration.ece_ci
    rows.append(
        (
            "Overconfident answers",
            "`confidence_shift=0.18`",
            "large positive ECE, `overconfident`",
            f"ECE {b.calibration.ece:.3f} [{lo:.3f}, {hi:.3f}]",
            f"{b.calibration.direction} → {b.calibration.verdict}",
        )
    )

    # 4. Canonical-order memorization.
    run = generate_run(n_questions=N, n_variants=4, seed=SEED, canonical_bonus=0.12)
    b = run_audit(run, n_boot=N_BOOT)
    accs: dict[str, list[int]] = {}
    for r in run.responses:
        accs.setdefault(r.variant_id, [0, 0])
        accs[r.variant_id][0] += int(r.selected_index == r.gold_index)
        accs[r.variant_id][1] += 1
    canon = accs["canonical"][0] / accs["canonical"][1]
    shuffled = float(
        np.mean([v[0] / v[1] for name, v in accs.items() if name != "canonical"])
    )
    designed = -0.12 * 0.55  # know-branch share of the run
    rows.append(
        (
            "Canonical-order memorization",
            "`canonical_bonus=0.12`",
            f"{designed:+.3f} accuracy drop on shuffles",
            f"{shuffled - canon:+.3f} measured drop",
            f"McNemar p = {b.order.mcnemar_p:.0e} → systematic",
        )
    )

    print("| Injected bias | Generator knob | Designed effect | Recovered (95% CI) | Verdict |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
