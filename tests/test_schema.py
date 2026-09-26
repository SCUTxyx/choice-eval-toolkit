"""Schema roundtrip and validation."""

import pytest

from choice_eval.schema import EvalRun, Response, load_jsonl, save_jsonl


def _resp(**over):
    base = dict(
        question_id="q1",
        n_options=4,
        gold_index=2,
        selected_index=0,
        confidence=0.7,
        option_lengths=[10, 20, 30, 40],
        option_ids=["a", "b", "c", "d"],
        variant_id="default",
    )
    base.update(over)
    return Response(**base)


def test_validate_ok():
    assert _resp().validate() == []


def test_validate_catches_bad_fields():
    assert _resp(gold_index=4).validate()
    assert _resp(selected_index=9).validate()
    assert _resp(confidence=1.5).validate()
    assert _resp(option_lengths=[1, 2]).validate()


def test_jsonl_roundtrip(tmp_path):
    run = EvalRun(name="rt")
    run.add(_resp())
    run.add(_resp(question_id="q2", selected_index=None, confidence=None))
    path = tmp_path / "run.jsonl"
    save_jsonl(run, path)
    loaded = load_jsonl(path)  # name defaults to the file stem; JSONL carries no run name
    assert loaded.name == "run"
    assert len(loaded) == 2
    a, b = loaded.responses
    assert (a.question_id, a.gold_index, a.selected_index, a.confidence) == ("q1", 2, 0, 0.7)
    assert b.selected_index is None and b.confidence is None


def test_jsonl_rejects_garbage(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question_id": "q1", "n_options": 4}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        load_jsonl(path)

    path.write_text('{"question_id": "q1", "n_options": 4, "gold_index": 0, "selected_index": 1, "bogus": 1}\n')
    with pytest.raises(ValueError, match="unknown field"):
        load_jsonl(path)

    path.write_text("not json\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_jsonl(path)
