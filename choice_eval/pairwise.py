"""Pairwise (arena-style) preference audit.

Common in embodied evaluation: two candidate trajectories/plans/policies are
presented to a judge (human or LLM) and the judge picks the better one
(RoboArena-style). Four questions are audited:

1. *Slot preference* — is the content presented first picked more than half
   the time? (Binomial test vs 0.5, cluster bootstrap CI over pairs.)
   Confound: if content A is systematically presented first, slot preference
   is indistinguishable from content preference — the *slot balance* check
   (is A in slot 0 about half the time?) flags exactly this, mirroring the
   answer-key balance audit on the MCQ side.
2. *Swap consistency* — for pairs judged under BOTH presentation orders, does
   the same content win both times? Order-sensitive judges show up here.
3. *Length preference* — is the longer description/trajectory chosen more
   often than chance? (Within-pair statistic: P(chosen is longer) and the
   mean length advantage.)
4. *Corrected win rates* — per-content win rate split by presentation slot;
   with a balanced design this is the order-corrected leaderboard.

Hypothesis tests use one judgment per pair (independent rows); rates and CIs
use every judgment with pair-level cluster bootstrap, mirroring the MCQ
audits' design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .schema import PairwiseRun
from .stats import cluster_bootstrap_ci, cluster_bootstrap_ci_vec


@dataclass
class ContentWinRate:
    content_id: str
    appearances: int
    wins: int
    win_rate: float
    win_rate_as_first: float  # when this content was presented in slot 0
    win_rate_as_second: float


@dataclass
class PairwiseAudit:
    n_judgments: int
    n_pairs: int
    skipped: str | None
    # slot preference (model/judge side)
    n_decided: int
    pick_first_rate: float  # P(winner presented in slot 0)
    pick_first_ci: tuple[float, float]
    slot_p: float
    slot_effect: float  # |rate - 0.5|
    slot_verdict: str
    # dataset-side: is content assignment to slots balanced?
    slot_balance_rate: float  # P(content_a in slot 0), one row per pair
    slot_balance_p: float
    slot_balance: str  # balanced / imbalanced
    # swap consistency
    n_swap_pairs: int
    swap_consistency: float
    swap_ci: tuple[float, float]
    swap_verdict: str
    # length preference
    p_chosen_longer: float
    p_chosen_longer_ci: tuple[float, float]
    mean_length_advantage: float  # mean(len(chosen) - len(other)) in chars
    length_verdict: str
    # bookkeeping
    tie_rate: float
    content_win_rates: list[ContentWinRate]


_SLOT_BANDS = ((0.02, "none"), (0.05, "minor"), (0.10, "moderate"))


def _slot_verdict(effect: float, p_value: float, alpha: float = 0.01) -> str:
    if p_value >= alpha:
        return "none"
    for bound, name in _SLOT_BANDS:
        if effect < bound:
            return name
    return "severe"


def _length_verdict(rate: float, ci: tuple[float, float], p_value: float, alpha: float = 0.01) -> str:
    if p_value >= alpha:
        return "none"
    if ci[0] <= 0.5 <= ci[1]:
        # CI covers the coin flip: never claim more than a minor signal
        return "minor"
    effect = abs(rate - 0.5)
    if effect < 0.03:
        return "minor"
    if effect < 0.07:
        return "moderate"
    return "severe"


def audit_pairwise(
    run: PairwiseRun,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairwiseAudit:
    judgments = run.judgments
    n = len(judgments)
    pair_ids = [j.pair_id for j in judgments]
    pair_names = sorted(set(pair_ids))
    pair_code = {p: i for i, p in enumerate(pair_names)}

    slot_of_a = np.array([j.slot_of_a for j in judgments])
    selected = np.array([-1 if j.selected_slot is None else j.selected_slot for j in judgments])
    decided = np.array([j.selected_slot is not None for j in judgments])
    winner_is_a = np.full(n, np.nan)
    winner_is_a[decided] = (selected[decided] == slot_of_a[decided]).astype(float)
    lengths_ok = np.array([j.context_lengths is not None for j in judgments])
    len_a = np.full(n, np.nan)
    len_b = np.full(n, np.nan)
    for i, j in enumerate(judgments):
        if j.context_lengths is not None:
            len_a[i], len_b[i] = j.context_lengths

    # --- slot balance (dataset side): one row per pair, first judgment -------
    first_seen: set[int] = set()
    balance_mask = np.zeros(n, dtype=bool)
    for i, p in enumerate(pair_ids):
        code = pair_code[p]
        if code not in first_seen:
            first_seen.add(code)
            balance_mask[i] = True
    a_first = slot_of_a[balance_mask]  # 1 where content A sits in slot 0
    n_bal = int(len(a_first))  # number of pairs contributing to the check
    slot_balance_rate = float(a_first.mean()) if n_bal else float("nan")
    if n_bal:
        slot_balance_p = float(stats.binomtest(int(a_first.sum()), n_bal, 0.5).pvalue)
    else:
        slot_balance_p = float("nan")
    slot_balance = "unknown" if n_bal == 0 else ("imbalanced" if slot_balance_p < 0.01 else "balanced")

    # --- slot preference (judge side, decided rows) ---------------------------
    dec = decided
    n_decided = int(dec.sum())
    nan = float("nan")
    pick_first_rate = float("nan")
    pick_first_ci = (nan, nan)
    slot_p = nan
    slot_effect = nan
    slot_verdict = "not available"
    if n_decided:
        chosen_slot = selected[dec]
        wins_first = int((chosen_slot == 0).sum())
        pick_first_rate = wins_first / n_decided
        slot_p = float(stats.binomtest(wins_first, n_decided, 0.5).pvalue)
        slot_effect = abs(pick_first_rate - 0.5)
        slot_verdict = _slot_verdict(slot_effect, slot_p)
        pair_codes_dec = np.array([pair_code[p] for p in np.array(pair_ids, dtype=object)[dec]])
        ci = cluster_bootstrap_ci(
            lambda idx: float(np.mean(selected[dec][idx] == 0)), pair_codes_dec, n_boot, alpha, seed + 1
        )
        pick_first_ci = ci.as_tuple()

    # --- swap consistency (pairs judged under both slot orders) ---------------
    by_pair: dict[int, list[int]] = {}
    for i, p in enumerate(pair_ids):
        by_pair.setdefault(pair_code[p], []).append(i)

    swap_same: list[int] = []
    swap_pair_codes: list[int] = []
    for code, rows in by_pair.items():
        rows_by_order = {}
        for i in rows:
            rows_by_order.setdefault(slot_of_a[i], []).append(i)
        if 0 in rows_by_order and 1 in rows_by_order:
            for i in rows_by_order[0]:
                for jj in rows_by_order[1]:
                    if decided[i] and decided[jj]:
                        swap_same.append(int(winner_is_a[i] == winner_is_a[jj]))
                        swap_pair_codes.append(code)

    n_swap_pairs = len(swap_same)
    swap_consistency = float("nan")
    swap_ci = (nan, nan)
    swap_verdict = "not available"
    if n_swap_pairs:
        same = np.array(swap_same, dtype=float)
        swap_consistency = float(same.mean())
        swap_ci = cluster_bootstrap_ci(
            lambda idx: float(np.mean(same[idx])), np.array(swap_pair_codes), n_boot, alpha, seed + 2
        ).as_tuple()
        if swap_consistency >= 0.90:
            swap_verdict = "stable"
        elif swap_consistency >= 0.75:
            swap_verdict = "mostly stable"
        else:
            swap_verdict = "unstable"

    # --- length preference -----------------------------------------------------
    p_chosen_longer = nan
    p_longer_ci = (nan, nan)
    mean_advantage = nan
    length_verdict = "not available"
    len_usable = dec & lengths_ok & (len_a != len_b)
    if len_usable.sum():
        # winner content = a iff selected_slot == slot_of_a (slot_of_a varies
        # across presentation orders, so resolving by slot alone misaligns)
        winner_is_a_len = selected[len_usable] == slot_of_a[len_usable]
        chosen_len = np.where(winner_is_a_len, len_a[len_usable], len_b[len_usable])
        other_len = np.where(winner_is_a_len, len_b[len_usable], len_a[len_usable])
        chosen_longer = (chosen_len > other_len).astype(float)
        n_len = int(len_usable.sum())
        p_chosen_longer = float(chosen_longer.mean())
        length_p = float(stats.binomtest(int(chosen_longer.sum()), n_len, 0.5).pvalue)
        pair_codes_len = np.array([pair_code[p] for p in np.array(pair_ids, dtype=object)[len_usable]])
        # Contents recur across pairs, so judgments correlate through shared
        # contents too — cluster on the pair's first-listed content for a
        # (partially) content-level CI; see the report's method notes for the
        # attribution caveat this implies.
        content_codes = {}
        content_cluster = []
        for p, j in zip(np.array(pair_ids, dtype=object)[len_usable],
                        [judgments[i] for i in np.flatnonzero(len_usable)]):
            key = j.content_a_id
            if key not in content_codes:
                content_codes[key] = len(content_codes)
            content_cluster.append(content_codes[key])
        p_longer_ci = cluster_bootstrap_ci(
            lambda idx: float(np.mean(chosen_longer[idx])), np.array(content_cluster), n_boot, alpha, seed + 3
        ).as_tuple()
        length_verdict = _length_verdict(p_chosen_longer, p_longer_ci, length_p)
        mean_advantage = float((chosen_len - other_len).mean())

    # --- per-content corrected win rates ---------------------------------------
    wins: dict[str, list[float]] = {}
    wins_first: dict[str, list[float]] = {}   # samples where the content sat in slot 0
    wins_second: dict[str, list[float]] = {}  # samples where the content sat in slot 1
    for i, j in enumerate(judgments):
        if decided[i]:
            w_a = float(winner_is_a[i])
            entries = (
                (j.content_a_id, w_a, j.slot_of_a == 0),
                (j.content_b_id, 1.0 - w_a, j.slot_of_a == 1),
            )
            for cid, won, in_slot0 in entries:
                wins.setdefault(cid, []).append(won)
                if in_slot0:
                    wins_first.setdefault(cid, []).append(won)
                else:
                    wins_second.setdefault(cid, []).append(won)
        else:
            for cid in (j.content_a_id, j.content_b_id):
                wins.setdefault(cid, []).append(np.nan)

    content_win_rates = []
    for cid, bucket in wins.items():
        arr = np.asarray(bucket, dtype=float)
        decided_mask = ~np.isnan(arr)
        appearances = int(decided_mask.sum())
        if appearances == 0:
            continue
        first = np.asarray(wins_first.get(cid, []), dtype=float)
        second = np.asarray(wins_second.get(cid, []), dtype=float)
        content_win_rates.append(
            ContentWinRate(
                content_id=cid,
                appearances=appearances,
                wins=int(arr[decided_mask].sum()),
                win_rate=float(arr[decided_mask].mean()),
                win_rate_as_first=float(first.mean()) if len(first) else nan,
                win_rate_as_second=float(second.mean()) if len(second) else nan,
            )
        )
    content_win_rates.sort(key=lambda c: (-c.win_rate, -c.appearances))

    tie_rate = float(np.mean(~decided)) if n else nan

    skipped: str | None = None
    if n == 0:
        skipped = "no judgments"
    elif n_decided == 0:
        skipped = "every judgment is a tie/undecided; preference statistics unavailable"

    return PairwiseAudit(
        n_judgments=n,
        n_pairs=len(pair_names),
        skipped=skipped,
        n_decided=n_decided,
        pick_first_rate=pick_first_rate,
        pick_first_ci=pick_first_ci,
        slot_p=slot_p,
        slot_effect=slot_effect,
        slot_verdict=slot_verdict,
        slot_balance_rate=slot_balance_rate,
        slot_balance_p=slot_balance_p,
        slot_balance=slot_balance,
        n_swap_pairs=n_swap_pairs,
        swap_consistency=swap_consistency,
        swap_ci=swap_ci,
        swap_verdict=swap_verdict,
        p_chosen_longer=p_chosen_longer,
        p_chosen_longer_ci=p_longer_ci,
        mean_length_advantage=mean_advantage,
        length_verdict=length_verdict,
        tie_rate=tie_rate,
        content_win_rates=content_win_rates,
    )
