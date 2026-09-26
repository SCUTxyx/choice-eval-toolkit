"""Markdown report generation with figures."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .abstention import AbstentionAudit, audit_abstention  # noqa: E402
from .arrays import to_arrays  # noqa: E402
from .calibration import CalibrationAudit, audit_calibration  # noqa: E402
from .length import LengthAudit, audit_length  # noqa: E402
from .order import OrderAudit, audit_order  # noqa: E402
from .position import PositionAudit, audit_position  # noqa: E402
from .schema import EvalRun  # noqa: E402


@dataclass
class AuditBundle:
    run_name: str
    n_responses: int
    n_questions: int
    n_variants: int
    k: int
    abstain_rate: float
    accuracy: float
    position: PositionAudit
    length: LengthAudit
    order: OrderAudit
    calibration: CalibrationAudit
    abstention: AbstentionAudit


def run_audit(
    run: EvalRun,
    n_bins: int = 10,
    binning: str = "equal_width",
    n_boot: int = 1000,
    seed: int = 0,
    target_risk: float = 0.15,
) -> AuditBundle:
    arr = to_arrays(run)
    answered = arr.answered
    return AuditBundle(
        run_name=run.name,
        n_responses=len(run),
        n_questions=len(arr.qid_names),
        n_variants=len(arr.variant_names),
        k=arr.k,
        abstain_rate=float(np.mean(~answered)),
        accuracy=float(arr.correct[answered].mean()) if answered.any() else float("nan"),
        position=audit_position(run, n_boot=n_boot, seed=seed),
        length=audit_length(run, n_boot=n_boot, seed=seed),
        order=audit_order(run, n_boot=n_boot, seed=seed),
        calibration=audit_calibration(run, n_bins=n_bins, binning=binning, n_boot=n_boot, seed=seed),
        abstention=audit_abstention(run, target_risk=target_risk),
    )


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _pfmt(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 1e-6:
        return "< 1e-6"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def write_report(bundle: AuditBundle, out_dir: str | Path, title: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_figures(bundle, out)
    path = out / "report.md"
    path.write_text(_render(bundle, title), encoding="utf-8")
    return path


def _write_figures(b: AuditBundle, out: Path) -> None:
    pa, la, ca, ab = b.position, b.length, b.calibration, b.abstention

    # Position selection rates.
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(pa.k)
    yerr = np.array([pa.selection_rates - pa.selection_ci[:, 0], pa.selection_ci[:, 1] - pa.selection_rates])
    ax.bar(x, pa.selection_rates, yerr=yerr, capsize=4, color="#4C72B0", alpha=0.85)
    ax.axhline(1.0 / pa.k, color="crimson", ls="--", lw=1, label=f"uniform = {1.0 / pa.k:.3f}")
    ax.set_ylim(0, max(pa.selection_ci[:, 1].max() * 1.3, 0.35))  # headroom for legend
    ax.set_xticks(x, [chr(65 + i) for i in range(pa.k)])
    ax.set_xlabel("presented position")
    ax.set_ylabel("selection rate")
    ax.set_title("Selection rate by position")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_position.png", dpi=150)
    plt.close(fig)

    # Length rank selection rates.
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    yerr = np.array([la.selection_by_rank - la.selection_by_rank_ci[:, 0], la.selection_by_rank_ci[:, 1] - la.selection_by_rank])
    ax.bar(x, la.selection_by_rank, yerr=yerr, capsize=4, color="#55A868", alpha=0.85)
    ax.axhline(1.0 / la.k, color="crimson", ls="--", lw=1, label=f"uniform = {1.0 / la.k:.3f}")
    ax.set_ylim(0, max(la.selection_by_rank_ci[:, 1].max() * 1.3, 0.35))  # headroom for legend
    ax.set_xticks(x, ["shortest"] + [f"rank {i}" for i in range(1, la.k - 1)] + ["longest"])
    ax.set_ylabel("selection rate")
    ax.set_title("Selection rate by option length rank")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_length.png", dpi=150)
    plt.close(fig)

    # Reliability diagram.
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    centers = (ca.bin_edges[:-1] + ca.bin_edges[1:]) / 2
    width = np.diff(ca.bin_edges)
    ax.bar(centers, ca.bin_accuracy, width=width * 0.92, color="#4C72B0", alpha=0.85, label="accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_reliability.png", dpi=150)
    plt.close(fig)

    # Risk-coverage curve.
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(ab.coverage_curve, ab.risk_curve, color="#4C72B0", lw=1.8)
    ax.axhline(ab.target_risk, color="crimson", ls="--", lw=1, label=f"target risk = {ab.target_risk:.2f}")
    if not np.isnan(ab.suggested_threshold):
        ax.plot(ab.coverage_at_target, ab.risk_at_threshold, "o", color="crimson",
                label=f"threshold {ab.suggested_threshold:.2f} @ cov {ab.coverage_at_target:.2f}")
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("risk (error rate)")
    ax.set_title("Risk-coverage curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_risk_coverage.png", dpi=150)
    plt.close(fig)


def _render(b: AuditBundle, title: str | None) -> str:
    pa, la, oa, ca, ab = b.position, b.length, b.order, b.calibration, b.abstention
    k = b.k
    letters = [chr(65 + i) for i in range(k)]
    today = _dt.date.today().isoformat()
    title = title or f"Bias & Calibration Audit — {b.run_name}"

    L: list[str] = []
    L.append(f"# {title}")
    L.append("")
    L.append(
        f"*Generated {today} · {b.n_responses} responses · {b.n_questions} questions · "
        f"{b.n_variants} ordering(s) · {k} options · abstain/unparsable rate {b.abstain_rate:.1%} · "
        f"accuracy {b.accuracy:.1%}*"
    )
    L.append("")

    # ---- Summary -----------------------------------------------------------
    L.append("## Summary")
    L.append("")
    L.append("| Audit | Key statistic | Value | p-value | Verdict |")
    L.append("|---|---|---|---|---|")
    L.append(
        f"| Position bias | χ² GOF vs uniform (Cohen's w = {_fmt(pa.effect_w)}) "
        f"| χ² = {_fmt(pa.chi2, 1)} | {_pfmt(pa.p_value)} | **{pa.verdict}** |"
    )
    L.append(
        f"| Answer-key balance (dataset) | χ² GOF vs uniform (w = {_fmt(pa.gold_w)}) "
        f"| χ² = {_fmt(pa.gold_chi2, 1)} | {_pfmt(pa.gold_p)} | **{pa.gold_verdict}** |"
    )
    L.append(
        f"| Length bias | χ² GOF by length rank (w = {_fmt(la.rank_w)}) "
        f"| χ² = {_fmt(la.rank_chi2, 1)} | {_pfmt(la.rank_p)} | **{la.verdict}** |"
    )
    L.append(
        f"| Length artifact (dataset) | gold-at-length-rank GOF "
        f"| — | {_pfmt(la.gold_rank_p)} | **{la.gold_artifact}** |"
    )
    if oa.skipped:
        L.append(f"| Ordering consistency | — | — | — | *skipped: {oa.skipped}* |")
    else:
        L.append(
            f"| Ordering consistency | same content across orderings "
            f"| {_fmt(oa.content_consistency)} "
            f"[{_fmt(oa.content_ci[0])}, {_fmt(oa.content_ci[1])}] | — | **{oa.verdict}** |"
        )
        L.append(
            f"| Ordering × correctness | McNemar on paired correctness "
            f"| χ² = {_fmt(oa.mcnemar_chi2, 1)} | {_pfmt(oa.mcnemar_p)} | "
            f"**{'systematic direction' if oa.mcnemar_p < 0.01 else 'symmetric'}** |"
        )
    L.append(
        f"| Calibration | ECE ({ca.binning}, {len(ca.bin_edges) - 1} bins) "
        f"| {_fmt(ca.ece)} [{_fmt(ca.ece_ci[0])}, {_fmt(ca.ece_ci[1])}] | — | **{ca.verdict}** |"
    )
    L.append(
        f"| Selective prediction | AURC / E-AURC | {_fmt(ab.aurc)} / {_fmt(ab.e_aurc)} | — | — |"
    )
    L.append("")

    # ---- Position ----------------------------------------------------------
    L.append("## 1. Position / label bias")
    L.append("")
    L.append("Selection rates by presented position, with cluster-bootstrap 95% CIs "
             "(resampling questions, so re-ordered variants of one question move together):")
    L.append("")
    L.append("| Position | Selection rate | 95% CI | Excess vs other positions |")
    L.append("|---|---|---|---|")
    for c in range(k):
        lo, hi = pa.selection_ci[c]
        elo, ehi = pa.excess_ci[c]
        L.append(
            f"| {letters[c]} | {pa.selection_rates[c]:.3f} | [{lo:.3f}, {hi:.3f}] "
            f"| {pa.excess[c]:+.3f} [{elo:+.3f}, {ehi:+.3f}] |"
        )
    L.append("")
    L.append(
        f"χ²({k - 1}) = {pa.chi2:.1f}, p = {_pfmt(pa.p_value)}, Cohen's w = {pa.effect_w:.3f} "
        f"→ **{pa.verdict}**. First-position selection rate: {pa.first_rate:.3f} "
        f"(uniform would be {1.0 / k:.3f})."
    )
    L.append("")
    L.append("![Selection rate by position](fig_position.png)")
    L.append("")
    L.append("### Accuracy by gold position")
    L.append("")
    L.append(
        "If the model were position-blind, accuracy would not depend on where the correct "
        "answer sits (given a balanced key):"
    )
    L.append("")
    L.append("| Gold at | " + " | ".join(letters) + " |")
    L.append("|---|" + "---|" * k)
    L.append("| Accuracy | " + " | ".join(f"{a:.3f}" for a in pa.accuracy_by_position) + " |")
    L.append("")
    L.append(f"Homogeneity χ² p = {_pfmt(pa.accuracy_p)}.")
    L.append("")

    # ---- Dataset side ------------------------------------------------------
    L.append("## 2. Answer-key balance (dataset side)")
    L.append("")
    L.append("| Gold at | " + " | ".join(letters) + " |")
    L.append("|---|" + "---|" * k)
    L.append("| Share of questions | " + " | ".join(f"{a:.3f}" for a in pa.gold_distribution) + " |")
    L.append("")
    L.append(
        f"χ²({k - 1}) = {pa.gold_chi2:.1f}, p = {_pfmt(pa.gold_p)} → **{pa.gold_verdict}**. "
        "An imbalanced key inflates position bias: a model with a matching slot preference "
        "scores better than it deserves."
    )
    L.append("")

    # ---- Length ------------------------------------------------------------
    L.append("## 3. Length bias")
    L.append("")
    header = " | ".join(["shortest"] + [f"rank {i}" for i in range(1, k - 1)] + ["longest"]) if k > 2 else "shortest | longest"
    L.append("| Statistic | " + header + " |")
    L.append("|---|" + "---|" * k)
    L.append("| Selection rate | " + " | ".join(f"{a:.3f}" for a in la.selection_by_rank) + " |")
    L.append("| Gold share (dataset) | " + " | ".join(f"{a:.3f}" for a in la.gold_rank_distribution) + " |")
    L.append("")
    L.append(
        f"Selection by rank: χ²({k - 1}) = {la.rank_chi2:.1f}, p = {_pfmt(la.rank_p)}, w = {la.rank_w:.3f} "
        f"→ **{la.verdict}**. Mean length z-score of the selected option = {la.mean_z:+.3f} "
        f"[{la.mean_z_ci[0]:+.3f}, {la.mean_z_ci[1]:+.3f}], one-sample t-test p = {_pfmt(la.z_p_value)}. "
        f"Gold-at-rank balance: p = {_pfmt(la.gold_rank_p)} → **{la.gold_artifact}**."
    )
    L.append("")
    L.append("![Selection rate by length rank](fig_length.png)")
    L.append("")

    # ---- Order -------------------------------------------------------------
    L.append("## 4. Ordering consistency")
    L.append("")
    if oa.skipped:
        L.append(f"*Skipped — {oa.skipped}*")
    else:
        L.append(
            f"Across {oa.n_pairs} ordered pairs from {oa.n_questions} multi-variant questions: "
            f"same-answer-content rate = **{oa.content_consistency:.3f}** "
            f"[{oa.content_ci[0]:.3f}, {oa.content_ci[1]:.3f}] "
            f"(per-question chance level ≈ {oa.chance_consistency:.3f}); "
            f"same-correctness rate = {oa.correct_consistency:.3f}. "
            f"McNemar χ² = {oa.mcnemar_chi2:.1f}, p = {_pfmt(oa.mcnemar_p)} → "
            f"{'**systematic direction** (e.g. canonical-order advantage)' if oa.mcnemar_p < 0.01 else 'no systematic correctness direction'}."
        )
    L.append("")

    # ---- Calibration -------------------------------------------------------
    L.append("## 5. Confidence calibration")
    L.append("")
    L.append(
        f"Accuracy = {ca.accuracy:.3f}, mean stated confidence = {ca.mean_confidence:.3f} "
        f"→ model is **{ca.direction}**. ECE = **{ca.ece:.3f}** "
        f"[{ca.ece_ci[0]:.3f}, {ca.ece_ci[1]:.3f}], MCE = {ca.mce:.3f} → **{ca.verdict}** "
        f"({ca.binning} binning, {len(ca.bin_edges) - 1} bins, n = {ca.n})."
    )
    L.append("")
    L.append("![Reliability diagram](fig_reliability.png)")
    L.append("")

    # ---- Abstention --------------------------------------------------------
    L.append("## 6. Selective prediction (abstention)")
    L.append("")
    L.append(
        f"Sorting answers by stated confidence: AURC = {ab.aurc:.3f}, E-AURC = {ab.e_aurc:.3f}."
    )
    if ab.coverage_at_target < 0.05 and not np.isnan(ab.suggested_threshold):
        L.append("")
        L.append(
            f"**No useful operating point exists for risk ≤ {ab.target_risk:.0%}**: even the "
            f"most confident {ab.coverage_at_target:.1%} of answers still err at "
            f"{ab.risk_at_threshold:.1%}. Confidence is too miscalibrated to threshold on "
            "(see §5) — recalibrate before deploying selective prediction."
        )
    elif not np.isnan(ab.suggested_threshold):
        L.append("")
        L.append(
            f"To hold risk ≤ {ab.target_risk:.0%}, answer only when confidence ≥ "
            f"**{ab.suggested_threshold:.2f}** — coverage {ab.coverage_at_target:.1%} "
            f"(observed risk at that point {ab.risk_at_threshold:.1%})."
        )
    else:
        L.append("")
        L.append(
            f"No confidence level achieves risk ≤ {ab.target_risk:.0%} on this run — "
            "selective prediction cannot meet the target."
        )
    L.append("")
    L.append("![Risk-coverage curve](fig_risk_coverage.png)")
    L.append("")

    # ---- Recommendations ---------------------------------------------------
    L.append("## Recommendations")
    L.append("")
    recs: list[str] = []
    if pa.verdict != "none":
        worst = int(np.argmax(pa.excess))
        recs.append(
            f"Position bias detected (largest excess at **{letters[worst]}**, "
            f"{pa.excess[worst]:+.3f}). Report accuracy averaged over cyclic (or random) "
            "permutations of the options, or debias before comparing models."
        )
    if pa.gold_verdict != "none":
        recs.append("Re-balance the answer key across positions before drawing model conclusions.")
    if la.verdict != "none":
        recs.append(
            "Length bias detected: control for option length (e.g. stratify accuracy by the "
            "gold option's length rank) or permute option order by length."
        )
    if la.gold_artifact == "artifact":
        recs.append(
            "Dataset artifact: the gold answer correlates with option length. Fix the benchmark "
            "or models will exploit the heuristic rather than the content."
        )
    if not oa.skipped and oa.verdict == "unstable":
        recs.append(
            "Low ordering consistency: the reported score depends on which ordering you used. "
            "Average over orderings and report consistency alongside accuracy."
        )
    if not oa.skipped and oa.mcnemar_p < 0.01:
        recs.append(
            "Systematic correctness direction across orderings (often canonical-order "
            "memorization): treat single-order scores as optimistic."
        )
    if ca.verdict in ("moderately miscalibrated", "poorly calibrated"):
        rec = (
            f"Calibration is {ca.direction}: consider temperature scaling on held-out data."
        )
        if ab.coverage_at_target >= 0.05 and not np.isnan(ab.suggested_threshold):
            rec += (
                f" For risky downstream decisions, answer only when confidence ≥ "
                f"{ab.suggested_threshold:.2f} (risk ≤ {ab.target_risk:.0%})."
            )
        else:
            rec += " Confidence is too miscalibrated for thresholding — recalibrate first."
        recs.append(rec)
    if not recs:
        recs.append("No significant bias detected at α = 0.01. Single-order scores remain "
                    "advisable to double-check on a fresh shuffle.")
    L.extend(f"- {r}" for r in recs)
    L.append("")

    L.append("## Method notes")
    L.append("")
    L.append(
        "- All CIs are cluster bootstrap percentile intervals (B = 1000 by default) with "
        "questions as clusters.\n"
        "- χ² and t-tests use one ordering per question: re-ordered variants of the same "
        "question are strongly correlated, and testing on all rows would overstate "
        "significance. Rates and CIs use every response.\n"
        "- χ² tests use α = 0.01; effect size is Cohen's w (0.1 / 0.2 / 0.3 ≈ small / medium / large).\n"
        "- E-AURC is the excess risk-coverage area over the oracle ordering (all correct "
        "answers first); smaller is better, 0 is unattainable in practice.\n"
        "- Abstain/unparsable answers are excluded from position, length and calibration "
        "statistics and counted in the abstain rate.\n"
        "- Equal-width bins can be noisy at the extremes; pass `--binning equal_mass` for "
        "quantile bins."
    )
    return "\n".join(L) + "\n"
