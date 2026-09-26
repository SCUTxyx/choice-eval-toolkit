"""Command-line interface.

    choice-eval audit RESPONSES.jsonl --out report_dir/
    choice-eval demo --out examples/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .generators import generate_pairwise, generate_run
from .report import run_audit, run_pairwise_audit, write_pairwise_report, write_report
from .schema import load_any, load_jsonl, save_jsonl, save_pairwise_jsonl

BIASED_KWARGS = dict(
    position_attract={2: 0.30},  # option C attracts 30% of would-be errors
    length_attract=0.35,  # longest option attracts 35% of remaining would-be errors
    confidence_shift=0.18,  # stated confidence is ~0.18 too high
    variant_flip=0.50,  # answers re-rolled across orderings half the time
    canonical_bonus=0.12,  # canonical ordering is answered systematically better
)

PAIRWISE_BIASED_KWARGS = dict(
    discernment=0.65,  # weaker judge
    slot_pref=0.20,  # picks whatever is presented first 20% of the time
    length_pref=0.30,  # undecided judgments favor the longer description
    order_flip=0.40,  # verdicts re-rolled 40% of the time when order swaps
    tie_prob=0.08,
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
    kind, run = load_any(args.input)
    if kind == "pairwise":
        bundle = run_pairwise_audit(run, n_boot=args.bootstrap, seed=args.seed)
        out = write_pairwise_report(bundle, args.out, title=f"Pairwise Preference Audit — {run.name}")
        pa = bundle.pairwise
        print(f"audit complete: {run.name} (pairwise, {len(run)} judgments, {bundle.n_pairs} pairs)")
        print(f"  slot preference : {pa.slot_verdict} (p = {pa.slot_p:.2e}), design {pa.slot_balance}")
        print(f"  swap consistency: {pa.swap_verdict}"
              + (f" ({pa.swap_consistency:.3f})" if pa.n_swap_pairs else " (skipped)"))
        print(f"  length preference: {pa.length_verdict}")
        print(f"report: {out}")
        return 0

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


def cmd_demo_pairwise(args) -> int:
    out = Path(args.out)
    specs = [
        ("pairwise_clean", {}),
        ("pairwise_biased", PAIRWISE_BIASED_KWARGS),
    ]
    for name, kwargs in specs:
        run = generate_pairwise(
            n_pairs=args.n,
            swap_orders=args.variants,
            seed=args.seed,
            name=f"demo-pairwise-{name}",
            **kwargs,
        )
        data_file = out / f"{name}_pairwise.jsonl"
        save_pairwise_jsonl(run, data_file)
        bundle = run_pairwise_audit(run, n_boot=args.bootstrap, seed=args.seed)
        report = write_pairwise_report(bundle, out / name)
        pa = bundle.pairwise
        print(f"[{name}] slot {pa.slot_verdict} ({pa.pick_first_rate:.3f}) | "
              f"swap {pa.swap_verdict} ({pa.swap_consistency:.3f}) | "
              f"length {pa.length_verdict} ({pa.p_chosen_longer:.3f})")
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

    p_pair = sub.add_parser("demo-pairwise", parents=[common], help="generate synthetic arena-style pairwise runs and demo reports")
    p_pair.add_argument("--n", type=int, default=1000, help="pairs per run (default 1000)")
    p_pair.add_argument("--variants", type=int, default=2, help="presentation orders per pair (default 2)")
    p_pair.add_argument("--out", default="examples")
    p_pair.set_defaults(func=cmd_demo_pairwise)

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
