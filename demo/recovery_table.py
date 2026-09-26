"""Print the injected-vs-recovered table used in the README.

Each row injects ONE bias into an otherwise clean synthetic run (n = 6000
questions) and reports what the audit measured. CIs come from the audit's
cluster bootstrap (questions as clusters). Run:

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


def _excess(rates, slot):
    return rates[slot] - np.delete(rates, slot).mean()


def _length_excess_ci(run, slot, k, n_boot=N_BOOT, seed=1):
    """Cluster bootstrap CI for the excess of one length rank over the others."""
    from choice_eval.stats import cluster_bootstrap_draws

    arr = to_arrays(run)
    usable = arr.answered & ~np.isnan(arr.lengths).any(axis=1)
    lengths = arr.lengths[usable]
    sel = arr.sel[usable]
    ranks = np.argsort(np.argsort(lengths, axis=1, kind="stable"), axis=1)
    sel_rank = ranks[np.arange(len(sel)), sel]
    qid = arr.qid[usable]

    def stat(idx):
        r = np.bincount(sel_rank[idx], minlength=k) / len(idx)
        return r[slot] - np.delete(r, slot).mean()

    lo, hi = np.percentile(cluster_bootstrap_draws(stat, qid, n_boot, seed), [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    rows = []

    # 1. Position attractor at option C.
    run = generate_run(n_questions=N, seed=SEED, position_attract={2: 0.30})
    b = run_audit(run, n_boot=N_BOOT)
    exp = expected_selection_rates(b.accuracy, 4, position_attract={2: 0.30})
    lo, hi = b.position.excess_ci[2]
    rows.append(
        (
            "Position pull toward option C",
            "`position_attract={2: 0.30}`",
            f"{_excess(exp, 2):+.3f} excess at C",
            f"{_excess(b.position.selection_rates, 2):+.3f} [{lo:+.3f}, {hi:+.3f}]",
            f"p = {b.position.p_value:.0e} → {b.position.verdict}",
        )
    )

    # 2. Longest-option attractor.
    run = generate_run(n_questions=N, seed=SEED, length_attract=0.25)
    b = run_audit(run, n_boot=N_BOOT)
    exp = expected_selection_rates(b.accuracy, 4, length_attract=0.25, space="rank")
    lo, hi = _length_excess_ci(run, 3, 4)
    rows.append(
        (
            "Longest-option pull",
            "`length_attract=0.25`",
            f"{_excess(exp, 3):+.3f} excess at longest rank",
            f"{_excess(b.length.selection_by_rank, 3):+.3f} [{lo:+.3f}, {hi:+.3f}]",
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
    p_min = 1.0 / (N_BOOT + 1)
    p_str = f"< {p_min:.0e}" if b.order.direction_p <= p_min else f"= {b.order.direction_p:.0e}"
    rows.append(
        (
            "Canonical-order memorization",
            "`canonical_bonus=0.12`",
            f"{designed:+.3f} accuracy drop on shuffles",
            f"{shuffled - canon:+.3f} measured drop",
            f"permutation p {p_str} → {b.order.direction}",
        )
    )

    print("| Injected bias | Generator knob | Designed effect | Recovered (95% CI) | Verdict |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
