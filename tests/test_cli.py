"""CLI smoke tests."""

from choice_eval.cli import main


def test_demo_command(tmp_path):
    out = tmp_path / "examples"
    rc = main(["demo", "--out", str(out), "--n", "150", "--variants", "2", "--bootstrap", "100"])
    assert rc == 0
    for name in ("clean", "biased"):
        assert (out / name / "report.md").exists()
        assert (out / f"{name}_responses.jsonl").exists()


def test_audit_command(tmp_path):
    out = tmp_path / "examples"
    assert main(["demo", "--out", str(out), "--n", "100", "--variants", "2", "--bootstrap", "100"]) == 0
    rc = main(
        [
            "audit", str(out / "biased_responses.jsonl"),
            "--out", str(tmp_path / "rep"),
            "--bootstrap", "100",
        ]
    )
    assert rc == 0
    assert (tmp_path / "rep" / "report.md").exists()


def test_audit_rejects_bad_input(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"n_options": 4}\n', encoding="utf-8")
    rc = main(["audit", str(bad), "--out", str(tmp_path / "rep")])
    assert rc == 2
