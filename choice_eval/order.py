"""Ordering-consistency audit (requires re-ordered variants of the same question).

For every question asked under two or more option orderings, this audit asks:

* *Content consistency* — did the model pick the same answer *content* both
  times? (Positions are matched through ``option_ids``; a model that picks
  "whatever sits at position C" is inconsistent by content.)
* *Correctness consistency* — did it get the same verdict both times? Discordant
  pairs are tested with McNemar's test, which detects a systematic direction,
  e.g. performing better when options appear in the dataset's canonical order
  (a common memorization signature).
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
    mcnemar_chi2: float
    mcnemar_p: float
    verdict: str
    skipped: str | None = None


def audit_order(
    run: EvalRun,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> OrderAudit:
    arr: RunArrays = to_arrays(run)

    skipped: str | None = None
    if len(arr.variant_names) < 2:
        skipped = "run contains a single ordering per question; re-run with shuffled options to enable this audit"
    elif any(oid is None for oid in arr.option_ids):
        skipped = "records lack option_ids; content identity across orderings cannot be established"

    by_q: dict[int, list[int]] = {}
    for i in range(arr.n):
        if skipped:
            break
        by_q.setdefault(int(arr.qid[i]), []).append(i)

    pair_same: list[int] = []
    pair_same_correct: list[int] = []
    pair_b: list[int] = []  # first correct, second wrong
    pair_c: list[int] = []  # first wrong, second correct
    pair_qid: list[int] = []
    chance_terms: list[float] = []
    used_questions = 0

    if not skipped:
        for q, idxs in sorted(by_q.items()):
            idxs = sorted(idxs, key=lambda i: arr.variant[i])
            if len(idxs) < 2:
                continue
            used_questions += 1
            contents: list[str | None] = []
            for i in idxs:
                if arr.sel[i] >= 0 and arr.option_ids[i] is not None:
                    contents.append(arr.option_ids[i][arr.sel[i]])
                else:
                    contents.append(None)
            corrects = [bool(arr.correct[i]) for i in idxs]

            # Chance level for "two random picks of this question coincide".
            k = arr.k
            counts = np.ones(k)  # +1 smoothing
            for c in contents:
                if c is not None:
                    counts[arr.option_ids[idxs[0]].index(c)] += 1
            p = counts / counts.sum()
            chance_terms.append(float((p**2).sum()))

            for a, b in combinations(range(len(idxs)), 2):
                ca, cb = contents[a], contents[b]
                if ca is not None and cb is not None:
                    pair_same.append(int(ca == cb))
                    pair_qid.append(q)
                    if corrects[a] and not corrects[b]:
                        pair_b.append(1)
                    elif corrects[b] and not corrects[a]:
                        pair_c.append(1)
                    pair_same_correct.append(int(corrects[a] == corrects[b]))

    if skipped or not pair_same:
        return OrderAudit(
            n_questions=0,
            n_pairs=0,
            content_consistency=float("nan"),
            content_ci=(float("nan"), float("nan")),
            chance_consistency=float("nan"),
            correct_consistency=float("nan"),
            mcnemar_chi2=float("nan"),
            mcnemar_p=float("nan"),
            verdict="not available",
            skipped=skipped or "no question has two usable orderings",
        )

    same = np.array(pair_same)
    b, c = sum(pair_b), sum(pair_c)
    chi2, p_mcnemar = mcnemar_exact(b, c)
    consistency = float(same.mean())
    ci = cluster_bootstrap_ci(
        lambda idx: float(np.mean(same[idx])), np.array(pair_qid), n_boot, alpha, seed
    )

    if consistency >= 0.95:
        verdict = "stable"
    elif consistency >= 0.85:
        verdict = "mostly stable"
    else:
        verdict = "unstable"

    return OrderAudit(
        n_questions=used_questions,
        n_pairs=len(same),
        content_consistency=consistency,
        content_ci=ci.as_tuple(),
        chance_consistency=float(np.mean(chance_terms)) if chance_terms else float("nan"),
        correct_consistency=float(np.mean(pair_same_correct)),
        mcnemar_chi2=chi2,
        mcnemar_p=p_mcnemar,
        verdict=verdict,
    )
