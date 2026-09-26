"""Synthetic evaluation runs with *injectable, known* biases.

This module is the toolkit's built-in exam paper: every bias is injected with an
explicit magnitude, so the audit functions can be checked against ground truth
(see ``tests/test_recovery.py``).

Generative model per question
-----------------------------
1. The question has ``k`` contents with fixed (content-level) option lengths.
2. The model either *knows* the answer (prob ``p_know``) or falls back to a
   misconception-anchored guess (correct with prob ``p_guess_correct``).
3. Each question is presented in ``n_variants`` random option orderings
   (variant 0 is the canonical order).
4. A wrong answer is drawn, in priority order, from:
   position attractors (``position_attract``) -> the longest wrong option
   (``length_attract``) -> the misconception (``misconception_stick``) ->
   uniform over wrong options.
5. Confidence is drawn around the branch's correctness probability; adding
   ``confidence_shift`` makes the run over/under-confident by construction.
6. Correctness is shared across orderings of the same question (the model
   either solves the question or it does not). ``variant_flip`` re-draws a
   variant's correctness with probability ``flip``; ``canonical_bonus``
   lowers the correctness probability of every non-canonical ordering
   independently (a memorization signature that McNemar's test picks up).

Closed-form effects: see :func:`expected_selection_rates`, which the recovery
tests (``tests/test_bias_audits.py``) compare the audits against.
"""

from __future__ import annotations

import numpy as np

from .schema import EvalRun, Response


def generate_run(
    n_questions: int = 1200,
    k: int = 4,
    n_variants: int = 1,
    seed: int = 0,
    p_know: float = 0.55,
    p_guess_correct: float = 0.30,
    misconception_stick: float = 0.80,
    position_attract: dict[int, float] | None = None,
    offset_attract: dict[int, float] | None = None,
    length_attract: float = 0.0,
    variant_flip: float = 0.0,
    canonical_bonus: float = 0.0,
    confidence_shift: float = 0.0,
    gold_long_bias: float = 0.0,
    abstain_prob: float = 0.0,
    name: str = "synthetic",
) -> EvalRun:
    """Generate an :class:`EvalRun` with the requested biases injected.

    ``offset_attract`` maps a *relative* position offset ``d`` (the option
    ``d`` slots after the gold one, wrapping around) to the probability that a
    wrong answer lands there — a relative-position preference that the
    gold-offset conditional test uniquely identifies.
    """
    if not 0.0 <= p_know <= 1.0:
        raise ValueError("p_know must be in [0, 1]")
    if position_attract and sum(position_attract.values()) > 0.95:
        raise ValueError("total position attractor mass must be <= 0.95")

    rng = np.random.default_rng(seed)
    run = EvalRun(name=name)
    if offset_attract:
        if any(d == 0 for d in offset_attract) or any(d < 0 or d >= k for d in offset_attract):
            raise ValueError("offset_attract keys must be non-zero offsets in 1..k-1")
        if sum(offset_attract.values()) > 0.95:
            raise ValueError("total offset attractor mass must be <= 0.95")

    for i in range(n_questions):
        qid = f"q{i:05d}"
        option_ids = [f"{qid}_opt{j}" for j in range(k)]
        # Content-level lengths are fixed across orderings (lognormal, ~10-90 chars).
        lengths_content = np.round(rng.lognormal(3.2, 0.45, size=k)).astype(int)

        if rng.random() < gold_long_bias:
            gold_content = int(np.argmax(lengths_content))
        else:
            gold_content = int(rng.integers(k))

        know = rng.random() < p_know
        p_correct = 0.88 if know else p_guess_correct
        # Whether the model solves this question is a property of the
        # (question, model) pair, so the correctness draw is shared across
        # orderings; flips and canonical-order effects re-draw per variant.
        base_correct = bool(rng.random() < p_correct)

        wrong_contents = [c for c in range(k) if c != gold_content]
        misconception = int(rng.choice(wrong_contents)) if wrong_contents else None

        conf_mean = float(np.clip(p_correct + confidence_shift, 0.01, 0.99))

        for v in range(n_variants):
            order = np.arange(k) if v == 0 else rng.permutation(k)
            content_at_pos = order  # position p presents content order[p]
            pos_of_content = np.argsort(order)

            gold_index = int(pos_of_content[gold_content])

            if v == 0:
                correct = base_correct
            elif know and canonical_bonus > 0.0:
                # Memorization signature: non-canonical orderings are answered
                # independently, and systematically worse.
                correct = rng.random() < (p_correct - canonical_bonus)
            elif rng.random() < variant_flip:
                correct = rng.random() < p_correct
            else:
                correct = base_correct

            if correct:
                selected_content = gold_content
            else:
                selected_content = _pick_wrong_content(
                    rng, k, gold_content, pos_of_content, content_at_pos,
                    position_attract, offset_attract, length_attract,
                    misconception_stick, misconception, lengths_content,
                    int(gold_index),
                )

            if rng.random() < abstain_prob:
                selected_index = None
                confidence = None
            else:
                selected_index = int(pos_of_content[selected_content])
                confidence = float(np.clip(conf_mean + rng.normal(0.0, 0.05), 0.01, 0.99))

            run.add(
                Response(
                    question_id=qid,
                    n_options=k,
                    gold_index=gold_index,
                    selected_index=selected_index,
                    confidence=confidence,
                    option_lengths=[int(x) for x in lengths_content[content_at_pos]],
                    option_ids=[option_ids[c] for c in content_at_pos],
                    variant_id="canonical" if v == 0 else f"shuffle_{v}",
                )
            )
    return run


def expected_offset_rates(
    k: int = 4,
    offset_attract: dict[int, float] | None = None,
) -> np.ndarray:
    """Closed-form P(selected = gold + d (mod k) | wrong) under the generative model.

    With no offset attractor this is uniform over the k-1 non-zero offsets.
    With attractor mass ``b_d`` at offset ``d``:
        offset[d] = b_d + (1 - sum(b)) / (k - 1)
    This conditional distribution is independent of the answer-key balance and
    of model accuracy — it is the ground truth the gold-offset test is graded
    against.
    """
    offsets = np.full(k - 1, 1.0 / (k - 1))
    if offset_attract:
        rest = 1.0 - sum(offset_attract.values())
        offsets[:] = rest / (k - 1)  # non-attracted offsets share the remainder
        for d, b in offset_attract.items():
            offsets[d - 1] = b + rest / (k - 1)
    return offsets


def expected_selection_rates(
    accuracy: float,
    k: int = 4,
    position_attract: dict[int, float] | None = None,
    length_attract: float = 0.0,
    space: str = "position",
) -> np.ndarray:
    """Closed-form selection-rate vector implied by the generative model.

    ``rates[r] = accuracy/k + (1-accuracy) * q[r]`` where ``q[r]`` is the
    probability of landing on slot ``r`` given a wrong answer, marginalized
    over the uniform gold placement. ``space="position"`` indexes presented
    positions; ``space="rank"`` indexes option length ranks (0 = shortest).
    This is the ground truth the recovery tests compare the audits against.

    The two attractors live in different spaces: a position attractor is a
    spike in position space but spreads uniformly in rank space (the content
    sitting at the attracted position has a random length), and vice versa.
    The length attractor targets the longest *wrong* option, so when the gold
    answer is itself the longest, part of its mass lands on rank ``k-2``.
    """
    if space not in ("position", "rank"):
        raise ValueError("space must be 'position' or 'rank'")
    if position_attract and len(position_attract) != 1:
        raise ValueError("closed form supports a single position attractor")
    c, b = next(iter(position_attract.items())) if position_attract else (None, 0.0)

    q = np.zeros(k)
    for gp in range(k):  # gold position (uniform)
        for gr in range(k):  # gold length rank (uniform, independent)
            w = 1.0 / (k * k)
            pa = b if (position_attract and c != gp) else 0.0
            rest = 1.0 - pa
            if space == "position":
                # attractor mass lands exactly on position c
                q[c] += w * pa
                # length attractor: the longest wrong content sits at a
                # uniformly random non-gold position -> invisible in this space
                # misconception / uniform: uniform over non-gold positions
                for r in range(k):
                    if r != gp:
                        q[r] += w * rest / (k - 1)
            else:
                # position attractor: content at position c has a uniform
                # non-gold rank
                if pa > 0:
                    for r in range(k):
                        if r != gr:
                            q[r] += w * pa / (k - 1)
                # length attractor: rank k-1, or k-2 when gold is longest
                if length_attract > 0:
                    q[(k - 1) if gr != (k - 1) else (k - 2)] += w * rest * length_attract
                # misconception / uniform: uniform over non-gold ranks
                for r in range(k):
                    if r != gr:
                        q[r] += w * rest * (1 - length_attract) / (k - 1)
    return accuracy / k + (1 - accuracy) * q


def _pick_wrong_content(
    rng,
    k: int,
    gold_content: int,
    pos_of_content: np.ndarray,
    content_at_pos: np.ndarray,
    position_attract: dict[int, float] | None,
    offset_attract: dict[int, float] | None,
    length_attract: float,
    stick: float,
    misconception: int | None,
    lengths_content: np.ndarray,
    gold_pos: int,
) -> int:
    """Draw a wrong *content* under the priority: offset attractor > position attractor > longest > misconception > uniform."""
    wrong_contents = [c for c in range(k) if c != gold_content]

    # Relative-position attractor: target slot is gold_pos + d (always holds a
    # wrong content, since d != 0), so it is always actionable.
    if offset_attract:
        total = sum(offset_attract.values())
        if rng.random() < total:
            x = rng.random() * total
            cum = 0.0
            for d, b in sorted(offset_attract.items()):
                cum += b
                if x < cum:
                    return int(content_at_pos[(gold_pos + d) % k])

    # A position attractor is actionable only when a *wrong* content sits there.
    actionable = [(p, b) for p, b in sorted((position_attract or {}).items()) if p != gold_pos]
    total = sum(b for _, b in actionable)
    if actionable and rng.random() < total:
        x = rng.random() * total
        cum = 0.0
        for p, b in actionable:
            cum += b
            if x < cum:
                return int(content_at_pos[p])

    if length_attract > 0.0 and rng.random() < length_attract:
        return max(wrong_contents, key=lambda c: int(lengths_content[c]))

    if misconception is not None and rng.random() < stick:
        return misconception

    return wrong_contents[int(rng.integers(len(wrong_contents)))]
