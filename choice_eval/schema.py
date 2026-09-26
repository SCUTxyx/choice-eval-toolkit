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


def load_jsonl(path: str | Path, name: str | None = None) -> EvalRun:
    """Load a run from a JSONL file; one line per response."""
    run = EvalRun(name=name or Path(path).stem)
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            missing = [f for f in REQUIRED_FIELDS if f not in record]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing required field(s) {missing}")
            unknown = [f for f in record if f not in REQUIRED_FIELDS + OPTIONAL_FIELDS]
            if unknown:
                raise ValueError(f"{path}:{lineno}: unknown field(s) {unknown}")
            run.add(
                Response(
                    question_id=str(record["question_id"]),
                    n_options=int(record["n_options"]),
                    gold_index=int(record["gold_index"]),
                    selected_index=None if record["selected_index"] is None else int(record["selected_index"]),
                    confidence=None if record.get("confidence") is None else float(record["confidence"]),
                    option_lengths=record.get("option_lengths"),
                    option_ids=record.get("option_ids"),
                    variant_id=str(record.get("variant_id", "default")),
                )
            )
    if not run.responses:
        raise ValueError(f"{path}: no responses found")
    return run


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
