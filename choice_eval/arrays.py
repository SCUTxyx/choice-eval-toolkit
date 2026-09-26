"""Convert an :class:`EvalRun` into aligned numpy arrays for the audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import EvalRun


@dataclass
class RunArrays:
    qid: np.ndarray  # integer question code, one entry per response
    qid_names: list[str]
    variant: np.ndarray  # integer variant code
    variant_names: list[str]
    k: int
    sel: np.ndarray  # selected position, -1 = abstain/unparsable
    gold: np.ndarray  # gold position per response
    conf: np.ndarray  # confidence, nan = missing
    lengths: np.ndarray  # (n, k) option lengths in presentation order, nan rows = missing
    option_ids: list  # per response: list[str] | None
    correct: np.ndarray  # 1/0, abstains count as incorrect
    answered: np.ndarray  # bool mask

    @property
    def n(self) -> int:
        return len(self.sel)


def to_arrays(run: EvalRun) -> RunArrays:
    responses = run.responses
    ks = {r.n_options for r in responses}
    if len(ks) != 1:
        raise ValueError(
            f"run mixes different option counts {sorted(ks)}; "
            "the audits expect a fixed number of options per run"
        )
    k = ks.pop()
    if k < 2:
        raise ValueError(
            f"n_options = {k}: audits need at least two options per question"
        )

    seen: set[tuple[str, str]] = set()
    for r in responses:
        key = (r.question_id, r.variant_id)
        if key in seen:
            raise ValueError(
                f"duplicate (question_id, variant_id) pair {key!r}: each question "
                "must appear at most once per ordering (for repeated samples of the "
                "same ordering, aggregate first or use distinct variant ids)"
            )
        seen.add(key)

    q_names = sorted({r.question_id for r in responses})
    q_code = {q: i for i, q in enumerate(q_names)}
    v_names = sorted({r.variant_id for r in responses})
    v_code = {v: i for i, v in enumerate(v_names)}

    n = len(responses)
    sel = np.full(n, -1, dtype=int)
    gold = np.empty(n, dtype=int)
    conf = np.full(n, np.nan)
    lengths = np.full((n, k), np.nan)
    option_ids: list = [None] * n
    correct = np.zeros(n, dtype=int)
    answered = np.zeros(n, dtype=bool)

    for i, r in enumerate(responses):
        gold[i] = r.gold_index
        if r.selected_index is not None:
            sel[i] = r.selected_index
            answered[i] = True
            correct[i] = int(r.selected_index == r.gold_index)
        if r.confidence is not None:
            conf[i] = r.confidence
        if r.option_lengths is not None:
            lengths[i] = r.option_lengths
        option_ids[i] = r.option_ids

    return RunArrays(
        qid=np.array([q_code[r.question_id] for r in responses]),
        qid_names=q_names,
        variant=np.array([v_code[r.variant_id] for r in responses]),
        variant_names=v_names,
        k=k,
        sel=sel,
        gold=gold,
        conf=conf,
        lengths=lengths,
        option_ids=option_ids,
        correct=correct,
        answered=answered,
    )


def first_variant_mask(arr: RunArrays) -> np.ndarray:
    """One row per question: the first variant encountered (for dataset-side audits)."""
    seen: set[int] = set()
    mask = np.zeros(arr.n, dtype=bool)
    for i in range(arr.n):
        q = int(arr.qid[i])
        if q not in seen:
            seen.add(q)
            mask[i] = True
    return mask
