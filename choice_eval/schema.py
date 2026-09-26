"""Data model and JSONL I/O for choice-mode evaluation records.

One ``Response`` is one (question, option-ordering) observation: the model saw
the question with options in some presentation order and picked one slot.
Option *identity* travels in ``option_ids`` so that answers given under
different orderings of the same question can be compared by content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_FIELDS = ("question_id", "n_options", "gold_index", "selected_index")
OPTIONAL_FIELDS = ("confidence", "option_lengths", "option_ids", "variant_id")

PAIR_REQUIRED_FIELDS = ("pair_id", "content_a_id", "content_b_id", "slot_of_a", "selected_slot")
PAIR_OPTIONAL_FIELDS = ("confidence", "context_lengths", "variant_id")
PAIR_FIELDS = set(PAIR_REQUIRED_FIELDS + PAIR_OPTIONAL_FIELDS)


@dataclass
class Response:
    """A single observed answer under one presentation order."""

    question_id: str
    n_options: int
    gold_index: int  # position of the correct option in THIS presentation order
    selected_index: int | None  # position chosen; None = abstain / unparsable
    confidence: float | None = None  # stated probability for the selected option, in [0, 1]
    option_lengths: list[int] | None = None  # character length of each presented option
    option_ids: list[str] | None = None  # content identity of each presented option
    variant_id: str = "default"  # which reordering of the question this is

    def validate(self) -> list[str]:
        errors = []
        if not 0 <= self.gold_index < self.n_options:
            errors.append(f"gold_index {self.gold_index} out of range for {self.n_options} options")
        if self.selected_index is not None and not 0 <= self.selected_index < self.n_options:
            errors.append(f"selected_index {self.selected_index} out of range")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            errors.append(f"confidence {self.confidence} outside [0, 1]")
        for name in ("option_lengths", "option_ids"):
            seq = getattr(self, name)
            if seq is not None and len(seq) != self.n_options:
                errors.append(f"{name} has {len(seq)} entries, expected {self.n_options}")
        return errors


@dataclass
class EvalRun:
    """A collection of responses from one evaluation (possibly many orderings)."""

    name: str = "run"
    responses: list[Response] = field(default_factory=list)

    def add(self, response: Response) -> None:
        errors = response.validate()
        if errors:
            raise ValueError(f"invalid response {response.question_id!r}: {'; '.join(errors)}")
        self.responses.append(response)

    def __len__(self) -> int:
        return len(self.responses)


def _as_int(value, field: str, lineno: int) -> int:
    """Strict integer coercion: booleans and fractional values are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"line {lineno}: {field} must be an integer, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"line {lineno}: {field} must be an integer, got {value!r}")
    return int(value)


def _as_float(value, field: str, lineno: int) -> float:
    """Strict float coercion: booleans and numeric strings are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"line {lineno}: {field} must be a number, got {value!r}")
    return float(value)


def load_jsonl(path: str | Path, name: str | None = None) -> EvalRun:
    """Load a run from a JSONL file; one line per response.

    Numeric fields are strictly typed: fractional indices, booleans and
    stringly-typed numbers are rejected with a line-precise message rather
    than silently coerced (a truncated ``gold_index`` silently moves the
    answer key; a boolean confidence reads as 0% or 100%).
    """
    run = EvalRun(name=name or Path(path).stem)
    with open(path, "r", encoding="utf-8-sig") as fh:  # utf-8-sig tolerates a BOM
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
            missing = [f for f in REQUIRED_FIELDS if f not in record]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing required field(s) {missing}")
            unknown = [f for f in record if f not in REQUIRED_FIELDS + OPTIONAL_FIELDS]
            if unknown:
                raise ValueError(f"{path}:{lineno}: unknown field(s) {unknown}")
            gold_index = _as_int(record["gold_index"], "gold_index", lineno)
            selected = record["selected_index"]
            selected_index = None if selected is None else _as_int(selected, "selected_index", lineno)
            confidence = record.get("confidence")
            if confidence is not None:
                confidence = _as_float(confidence, "confidence", lineno)
            n_options = _as_int(record["n_options"], "n_options", lineno)
            run.add(
                Response(
                    question_id=str(record["question_id"]),
                    n_options=n_options,
                    gold_index=gold_index,
                    selected_index=selected_index,
                    confidence=confidence,
                    option_lengths=record.get("option_lengths"),
                    option_ids=record.get("option_ids"),
                    variant_id=str(record.get("variant_id", "default")),
                )
            )
    if not run.responses:
        raise ValueError(f"{path}: no responses found")
    return run


@dataclass
class PairJudgment:
    """One pairwise (arena-style) judgment: which of two contents is better?

    Typical embodied use: two candidate trajectories/plans judged by a human
    or an LLM judge, presented in some slot order. The same pair judged under
    both presentation orders unlocks the swap-consistency audit.
    """

    pair_id: str  # identifies the two contents being compared
    content_a_id: str
    content_b_id: str
    slot_of_a: int  # slot (0 = presented first, 1 = second) where content A appeared
    selected_slot: int | None  # slot of the winner; None = tie / undecided
    confidence: float | None = None  # judge's stated confidence in the decision
    context_lengths: list[int] | None = None  # [len(content_a), len(content_b)]
    variant_id: str = "default"  # presentation-order label

    def validate(self) -> list[str]:
        errors = []
        if self.slot_of_a not in (0, 1):
            errors.append(f"slot_of_a must be 0 or 1, got {self.slot_of_a}")
        if self.selected_slot is not None and self.selected_slot not in (0, 1):
            errors.append(f"selected_slot must be 0, 1 or null, got {self.selected_slot}")
        if self.content_a_id == self.content_b_id:
            errors.append("content_a_id and content_b_id must differ")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            errors.append(f"confidence {self.confidence} outside [0, 1]")
        if self.context_lengths is not None and len(self.context_lengths) != 2:
            errors.append("context_lengths must have exactly 2 entries [len_a, len_b]")
        return errors

    @property
    def winner_content(self) -> str | None:
        """Content id of the winner, or None for ties/undecided."""
        if self.selected_slot is None:
            return None
        return self.content_a_id if self.selected_slot == self.slot_of_a else self.content_b_id


@dataclass
class PairwiseRun:
    """A collection of pairwise judgments (possibly over both presentation orders)."""

    name: str = "pairwise"
    judgments: list[PairJudgment] = field(default_factory=list)

    def add(self, judgment: PairJudgment) -> None:
        errors = judgment.validate()
        if errors:
            raise ValueError(f"invalid judgment {judgment.pair_id!r}: {'; '.join(errors)}")
        self.judgments.append(judgment)

    def __len__(self) -> int:
        return len(self.judgments)


def _looks_pairwise(record: dict) -> bool:
    """A record is pairwise iff it carries pair_id (MCQ carries question_id)."""
    return "pair_id" in record


def load_pairwise_jsonl(path: str | Path, name: str | None = None) -> PairwiseRun:
    """Load a pairwise run from JSONL; strict numeric typing as in :func:`load_jsonl`."""
    run = PairwiseRun(name=name or Path(path).stem)
    with open(path, "r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
            if not _looks_pairwise(record):
                raise ValueError(
                    f"{path}:{lineno}: not a pairwise record (missing 'pair_id'); "
                    "use load_jsonl for multiple-choice logs"
                )
            missing = [f for f in PAIR_REQUIRED_FIELDS if f not in record]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing required field(s) {missing}")
            unknown = [f for f in record if f not in PAIR_FIELDS]
            if unknown:
                raise ValueError(f"{path}:{lineno}: unknown field(s) {unknown}")
            slot_of_a = _as_int(record["slot_of_a"], "slot_of_a", lineno)
            selected = record["selected_slot"]
            selected_slot = None if selected is None else _as_int(selected, "selected_slot", lineno)
            confidence = record.get("confidence")
            if confidence is not None:
                confidence = _as_float(confidence, "confidence", lineno)
            lengths = record.get("context_lengths")
            if lengths is not None:
                if not isinstance(lengths, list) or len(lengths) != 2:
                    raise ValueError(f"{path}:{lineno}: context_lengths must be a 2-element list")
                lengths = [_as_float(v, "context_lengths", lineno) for v in lengths]
            run.add(
                PairJudgment(
                    pair_id=str(record["pair_id"]),
                    content_a_id=str(record["content_a_id"]),
                    content_b_id=str(record["content_b_id"]),
                    slot_of_a=slot_of_a,
                    selected_slot=selected_slot,
                    confidence=confidence,
                    context_lengths=lengths,
                    variant_id=str(record.get("variant_id", "default")),
                )
            )
    if not run.judgments:
        raise ValueError(f"{path}: no judgments found")
    return run


def save_pairwise_jsonl(run: PairwiseRun, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for j in run.judgments:
            fh.write(
                json.dumps(
                    {
                        "pair_id": j.pair_id,
                        "content_a_id": j.content_a_id,
                        "content_b_id": j.content_b_id,
                        "slot_of_a": j.slot_of_a,
                        "selected_slot": j.selected_slot,
                        "confidence": j.confidence,
                        "context_lengths": j.context_lengths,
                        "variant_id": j.variant_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_any(path: str | Path, name: str | None = None):
    """Load a JSONL log, auto-detecting the format from the first record.

    Returns ``(kind, run)`` where kind is ``"mcq"`` or ``"pairwise"``.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if _looks_pairwise(record):
                return "pairwise", load_pairwise_jsonl(path, name)
            return "mcq", load_jsonl(path, name)
    raise ValueError(f"{path}: no records found")


def save_jsonl(run: EvalRun, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in run.responses:
            fh.write(
                json.dumps(
                    {
                        "question_id": r.question_id,
                        "n_options": r.n_options,
                        "gold_index": r.gold_index,
                        "selected_index": r.selected_index,
                        "confidence": r.confidence,
                        "option_lengths": r.option_lengths,
                        "option_ids": r.option_ids,
                        "variant_id": r.variant_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
