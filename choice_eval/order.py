"""Ordering-consistency audit (requires re-ordered variants of the same question).

For every question asked under two or more option orderings, this audit asks:

* *Content consistency* — did the model pick the same answer *content* both
  times? (Positions are matched through ``option_ids``; a model that picks
  "whatever sits at position C" is inconsistent by content.) Reported against
  the per-question chance level so multi-option runs aren't misread.
* *Correctness consistency* — did it get the same verdict both times?

Directional test — is the model systematically *better* under some orderings
(e.g. the dataset's canonical order, a memorization signature)? The familiar
McNemar statistic is reported, but pooling all C(V,2) pairs into the exact
McNemar test would be anti-conservative: pairs of the same question are
correlated. Instead the p-value comes from a **variant-label permutation
test**: under the null hypothesis, variant labels are exchangeable, so we
re-randomize which row plays which variant within each question and recompute
the asymmetry. This is valid for any number of orderings and reduces to exact
McNemar when each question has two.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .arrays import RunArrays, to_arrays
from .schema import EvalRun
from .stats import CI, cluster_bootstrap_ci, mcnemar_exact


@dataclass
class OrderAudit:
    n_questions: int
    n_pairs: int
    content_consistency: float
    content_ci: tuple[float, float]
    chance_consistency: float
    correct_consistency: float
    mcnemar_chi2: float  # familiar statistic on pooled discordant pairs
    mcnemar_p: float  # exact McNemar p (valid when each question has one pair)
    direction_p: float  # variant-permutation p-value (cluster-valid, any V)
    direction: str  # "systematic direction" / "symmetric"
    verdict: str
    skipped: str | None = None


def _pair_asymmetry(corr: np.ndarray, valid: np.ndarray, perm: np.ndarray) -> tuple[float, float, int]:
    """(b, c, n_pairs) for correctness pairs i<j under a within-question row order.

    ``corr`` is (Q, V) 0/1 correctness, ``valid`` marks real entries, ``perm``
    gives the within-question presentation order being evaluated.
    """
    cp = np.take_along_axis(corr, perm, axis=1)
    vp = np.take_along_axis(valid, perm, axis=1)
    b = 0
    c = 0
    n_pairs = 0
    for i in range(corr.shape[1] - 1):
        for j in range(i + 1, corr.shape[1]):
            m = vp[:, i] & vp[:, j]
            b += int(np.sum(m & (cp[:, i] == 1) & (cp[:, j] == 0)))
            c += int(np.sum(m & (cp[:, i] == 0) & (cp[:, j] == 1)))
            n_pairs += int(m.sum())
    return b, c, n_pairs


def audit_order(
    run: EvalRun,
    n_boot: int = 1000,
    n_perm: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> OrderAudit:
    arr: RunArrays = to_arrays(run)

    skipped: str | None = None
    if len(arr.variant_names) < 2:
        skipped = "run contains a single ordering per question; re-run with shuffled options to enable this audit"
    elif any(oid is None for oid in arr.option_ids):
        skipped = "records lack option_ids; content identity across orderings cannot be established"

    nan = float("nan")
    if skipped:
        return OrderAudit(
            n_questions=0, n_pairs=0,
            content_consistency=nan, content_ci=(nan, nan),
            chance_consistency=nan, correct_consistency=nan,
            mcnemar_chi2=nan, mcnemar_p=nan, direction_p=nan,
            direction="not available", verdict="not available", skipped=skipped,
        )

    # Group rows by question, ordered by variant code.
    by_q: dict[int, list[int]] = {}
    for i in range(arr.n):
        by_q.setdefault(int(arr.qid[i]), []).append(i)

    questions: list[list[int]] = []
    pair_same: list[int] = []
    pair_same_correct: list[int] = []
    pair_qid: list[int] = []
    chance_terms: list[float] = []

    for q, rows in sorted(by_q.items()):
        rows = sorted(rows, key=lambda i: arr.variant[i])
        if len(rows) < 2:
            continue
        questions.append(rows)

        contents: list[str | None] = []
        for i in rows:
            if arr.sel[i] >= 0 and arr.option_ids[i] is not None:
                contents.append(arr.option_ids[i][arr.sel[i]])
            else:
                contents.append(None)
        corrects = [int(arr.correct[i]) for i in rows]

        # Chance level for "two random picks of this question coincide".
        counts = np.ones(arr.k)  # +1 smoothing
        for c in contents:
            if c is not None:
                counts[arr.option_ids[rows[0]].index(c)] += 1
        p = counts / counts.sum()
        chance_terms.append(float((p**2).sum()))

        for a, b in combinations(range(len(rows)), 2):
            ca, cb = contents[a], contents[b]
            if ca is not None and cb is not None:
                pair_same.append(int(ca == cb))
                pair_qid.append(q)
                pair_same_correct.append(int(corrects[a] == corrects[b]))

    if not pair_same:
        return OrderAudit(
            n_questions=0, n_pairs=0,
            content_consistency=nan, content_ci=(nan, nan),
            chance_consistency=nan, correct_consistency=nan,
            mcnemar_chi2=nan, mcnemar_p=nan, direction_p=nan,
            direction="not available", verdict="not available",
            skipped="no question has two usable orderings",
        )

    same = np.array(pair_same)
    consistency = float(same.mean())
    ci = cluster_bootstrap_ci(
        lambda idx: float(np.mean(same[idx])), np.array(pair_qid), n_boot, alpha, seed
    )

    # Correctness matrix per question (abstains count as incorrect).
    v_max = max(len(rows) for rows in questions)
    q_n = len(questions)
    corr = np.zeros((q_n, v_max), dtype=np.int64)
    valid = np.zeros((q_n, v_max), dtype=bool)
    for qi, rows in enumerate(questions):
        for vi, r in enumerate(rows):
            corr[qi, vi] = arr.correct[r]
            valid[qi, vi] = True

    obs_b, obs_c, n_pairs = _pair_asymmetry(corr, valid, np.tile(np.arange(v_max), (q_n, 1)))
    chi2, p_exact = mcnemar_exact(obs_b, obs_c)
    obs_stat = (obs_b - obs_c) / max(n_pairs, 1)

    rng = np.random.default_rng(seed + 777)
    exceed = 0
    for _ in range(n_perm):
        keys = rng.random((q_n, v_max))
        keys[~valid] = 2.0  # unused slots sort last
        perm = np.argsort(keys, axis=1)
        pb, pc, _ = _pair_asymmetry(corr, valid, perm)
        if abs((pb - pc) / n_pairs) >= abs(obs_stat) - 1e-12:
            exceed += 1
    p_perm = (1 + exceed) / (n_perm + 1)

    if consistency >= 0.95:
        verdict = "stable"
    elif consistency >= 0.85:
        verdict = "mostly stable"
    else:
        verdict = "unstable"

    return OrderAudit(
        n_questions=len(questions),
        n_pairs=len(same),
        content_consistency=consistency,
        content_ci=ci.as_tuple(),
        chance_consistency=float(np.mean(chance_terms)) if chance_terms else nan,
        correct_consistency=float(np.mean(pair_same_correct)),
        mcnemar_chi2=chi2,
        mcnemar_p=p_exact,
        direction_p=p_perm,
        direction="systematic direction" if p_perm < 0.01 else "symmetric",
        verdict=verdict,
    )
