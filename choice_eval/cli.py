"""Command-line interface.

    choice-eval audit RESPONSES.jsonl --out report_dir/
    choice-eval demo --out examples/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .generators import generate_run
from .report import run_audit, write_report
from .schema import load_jsonl, save_jsonl

BIASED_KWARGS = dict(
    position_attract={2: 0.30},  # option C attracts 30% of would-be errors
    length_attract=0.35,  # longest option attracts 35% of remaining would-be errors
    confidence_shift=0.18,  # stated confidence is ~0.18 too high
    variant_flip=0.50,  # answers re-rolled across orderings half the time
    canonical_bonus=0.12,  # canonical ordering is answered systematically better
)


def _audit_run(run, args):
    return run_audit(
        run,
        n_bins=args.bins,
        binning=args.binning,
        n_boot=args.bootstrap,
        seed=args.seed,
        target_risk=args.target_risk,
    )


def _report_lines(bundle) -> list[str]:
    lines = [
        f"  position bias : {bundle.position.verdict if bundle.position.skipped is None else 'skipped'}"
        + (f" (p = {bundle.position.p_value:.2e})" if bundle.position.skipped is None else ""),
        f"  length bias   : {bundle.length.verdict if bundle.length.skipped is None else 'skipped'}"
        + (f" (p = {bundle.length.rank_p:.2e})" if bundle.length.skipped is None else ""),
        f"  order         : {bundle.order.verdict}",
        f"  calibration   : {bundle.calibration.verdict if bundle.calibration.skipped is None else 'skipped'}"
        + (f" (ECE = {bundle.calibration.ece:.3f})" if bundle.calibration.skipped is None else ""),
    ]
    return lines


def cmd_audit(args) -> int:
    run = load_jsonl(args.input)
    bundle = _audit_run(run, args)
    out = write_report(bundle, args.out, title=f"Bias & Calibration Audit — {run.name}")
    print(f"audit complete: {run.name} ({len(run)} responses)")
    for line in _report_lines(bundle):
        print(line)
    print(f"report: {out}")
    return 0


def cmd_demo(args) -> int:
    out = Path(args.out)
    specs = [
        ("clean", {}),
        ("biased", BIASED_KWARGS),
    ]
    for name, kwargs in specs:
        run = generate_run(
            n_questions=args.n,
            k=args.k,
            n_variants=args.variants,
            seed=args.seed,
            name=f"demo-{name}",
            **kwargs,
        )
        data_file = out / f"{name}_responses.jsonl"
        save_jsonl(run, data_file)
        bundle = _audit_run(run, args)
        report = write_report(bundle, out / name)
        print(f"[{name}] accuracy {bundle.accuracy:.1%} | "
              f"position {bundle.position.verdict} | length {bundle.length.verdict} | "
              f"order {bundle.order.verdict} | ECE {bundle.calibration.ece:.3f}")
        print(f"  data:   {data_file}")
        print(f"  report: {report}")
    print(f"\nBoth reports are under {out}/ — compare clean/ vs biased/ report.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="choice-eval",
        description="Audit multiple-choice evaluation runs for position/length/ordering bias "
        "and confidence calibration.",
    )
    parser.add_argument("--version", action="version", version=f"choice-eval-toolkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bins", type=int, default=10, help="calibration bins (default 10)")
    common.add_argument("--binning", choices=["equal_width", "equal_mass"], default="equal_width")
    common.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples (default 1000)")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--target-risk", type=float, default=0.15, help="risk target for abstention threshold")

    p_audit = sub.add_parser("audit", parents=[common], help="audit a JSONL evaluation log")
    p_audit.add_argument("input", help="JSONL file, one response per line (see README for the schema)")
    p_audit.add_argument("--out", default="report", help="output directory")
    p_audit.set_defaults(func=cmd_audit)

    p_demo = sub.add_parser("demo", parents=[common], help="generate synthetic runs and demo reports")
    p_demo.add_argument("--n", type=int, default=1200, help="questions per run (default 1200)")
    p_demo.add_argument("--k", type=int, default=4, help="options per question (default 4)")
    p_demo.add_argument("--variants", type=int, default=4, help="orderings per question (default 4)")
    p_demo.add_argument("--out", default="examples")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
