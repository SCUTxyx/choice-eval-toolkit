"""Markdown report generation with figures, plus machine-readable JSON export."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ._version import __version__  # noqa: E402
from .abstention import AbstentionAudit, audit_abstention  # noqa: E402
from .arrays import to_arrays  # noqa: E402
from .calibration import CalibrationAudit, audit_calibration  # noqa: E402
from .length import LengthAudit, audit_length  # noqa: E402
from .order import OrderAudit, audit_order  # noqa: E402
from .pairwise import PairwiseAudit, audit_pairwise  # noqa: E402
from .position import PositionAudit, audit_position  # noqa: E402
from .schema import EvalRun, PairwiseRun  # noqa: E402


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
    ordering_accuracy: list[tuple[str, int, float]] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def run_audit(
    run: EvalRun,
    n_bins: int = 10,
    binning: str = "equal_width",
    n_boot: int = 1000,
    n_perm: int = 1000,
    seed: int = 0,
    target_risk: float = 0.15,
) -> AuditBundle:
    """Run all audits and collect them into a reportable bundle."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if n_boot < 1 or n_perm < 1:
        raise ValueError("n_boot and n_perm must be >= 1")
    if not 0.0 < target_risk <= 1.0:
        raise ValueError("target_risk must be in (0, 1]")

    arr = to_arrays(run)
    answered = arr.answered

    ordering_accuracy: list[tuple[str, int, float]] = []
    if len(arr.variant_names) > 1:
        for v_code, v_name in enumerate(arr.variant_names):
            m = (arr.variant == v_code) & answered
            if m.sum():
                ordering_accuracy.append((v_name, int(m.sum()), float(arr.correct[m].mean())))

    params = {
        "toolkit_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_boot": n_boot,
        "n_perm": n_perm,
        "n_bins": n_bins,
        "binning": binning,
        "ci_alpha": 0.05,
        "test_alpha": 0.01,
        "target_risk": target_risk,
        "seed": seed,
    }

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
        order=audit_order(run, n_boot=n_boot, n_perm=n_perm, seed=seed),
        calibration=audit_calibration(run, n_bins=n_bins, binning=binning, n_boot=n_boot, seed=seed),
        abstention=audit_abstention(run, target_risk=target_risk),
        ordering_accuracy=ordering_accuracy,
        params=params,
    )


def bundle_to_dict(bundle: AuditBundle) -> dict:
    """JSON-safe dict of the whole bundle (NaN/Inf become null)."""
    return _clean(dataclasses.asdict(bundle))


def write_results_json(bundle: AuditBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle_to_dict(bundle), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _clean(obj):
    if isinstance(obj, np.ndarray):
        return [_clean(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return _clean(obj.item())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(x) for x in obj]
    return obj


def write_report(bundle: AuditBundle, out_dir: str | Path, title: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_figures(bundle, out)
    path = out / "report.md"
    path.write_text(_render(bundle, title), encoding="utf-8")
    write_results_json(bundle, out / "results.json")
    return path


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


def _slot_labels(k: int) -> list[str]:
    """A, B, ... Z, then L27, L28, ... for runs with more than 26 options."""
    return [chr(65 + i) if i < 26 else f"L{i + 1}" for i in range(k)]


def _write_figures(b: AuditBundle, out: Path) -> None:
    pa, la, ca, ab = b.position, b.length, b.calibration, b.abstention

    if pa.skipped is None:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        x = np.arange(pa.k)
        yerr = np.array([pa.selection_rates - pa.selection_ci[:, 0], pa.selection_ci[:, 1] - pa.selection_rates])
        ax.bar(x, pa.selection_rates, yerr=yerr, capsize=4, color="#4C72B0", alpha=0.85)
        ax.axhline(1.0 / pa.k, color="crimson", ls="--", lw=1, label=f"uniform = {1.0 / pa.k:.3f}")
        ax.set_ylim(0, max(pa.selection_ci[:, 1].max() * 1.3, 0.35))  # headroom for legend
        ax.set_xticks(x, _slot_labels(pa.k))
        ax.set_xlabel("presented position")
        ax.set_ylabel("selection rate")
        ax.set_title("Selection rate by position")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(out / "fig_position.png", dpi=150)
        plt.close(fig)

    if la.skipped is None:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        x = np.arange(la.k)
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

    if ca.skipped is None:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        centers = (ca.bin_edges[:-1] + ca.bin_edges[1:]) / 2
        width = np.diff(ca.bin_edges)
        with np.errstate(invalid="ignore"):
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

    if ab.skipped is None:
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
    letters = _slot_labels(k)
    today = _dt.date.today().isoformat()
    title = title or f"Bias & Calibration Audit — {b.run_name}"
    p = b.params

    L: list[str] = []
    L.append(f"# {title}")
    L.append("")
    L.append(
        f"*Generated {today} · {b.n_responses} responses · {b.n_questions} questions · "
        f"{b.n_variants} ordering(s) · {k} options · abstain/unparsable rate {b.abstain_rate:.1%} · "
        f"accuracy {b.accuracy:.1%}*"
    )
    L.append("")
    L.append(
        f"*Toolkit v{p.get('toolkit_version', '?')} · B = {p.get('n_boot')} bootstrap / "
        f"{p.get('n_perm')} permutations · seed {p.get('seed')} · {p.get('n_bins')} "
        f"{p.get('binning')} calibration bins · 95% CIs · tests at α = 0.01 · target risk "
        f"{p.get('target_risk'):.0%} — machine-readable results in `results.json`.*"
    )
    if b.n_questions < 30:
        L.append("")
        L.append(
            f"**Small-sample caution:** only {b.n_questions} questions — bootstrap CIs and "
            "χ² approximations are unstable at this size; treat every verdict as provisional."
        )
    L.append("")

    # ---- Summary -----------------------------------------------------------
    L.append("## Summary")
    L.append("")
    L.append("| Audit | Key statistic | Value | p-value | Verdict |")
    L.append("|---|---|---|---|---|")
    if pa.skipped:
        L.append(f"| Position bias | — | — | — | *skipped: {pa.skipped}* |")
    else:
        L.append(
            f"| Position bias (marginal) | χ² GOF vs uniform (Cohen's w = {_fmt(pa.effect_w)}) "
            f"| χ² = {_fmt(pa.chi2, 1)} | {_pfmt(pa.p_value)} | **{pa.marginal_verdict}** |"
        )
        off_p = "n/a" if pa.n_wrong == 0 else _pfmt(pa.offset_p)
        L.append(
            f"| Position bias (gold-offset, confound-free) | χ² GOF of (sel − gold) mod K among wrongs "
            f"(w = {_fmt(pa.offset_w)}) | χ² = {_fmt(pa.offset_chi2, 1)} | {off_p} | **{pa.offset_verdict}** |"
        )
    L.append(
        f"| Answer-key balance (dataset) | χ² GOF vs uniform (w = {_fmt(pa.gold_w)}) "
        f"| χ² = {_fmt(pa.gold_chi2, 1)} | {_pfmt(pa.gold_p)} | **{pa.gold_verdict}** |"
    )
    if la.skipped:
        L.append(f"| Length bias | — | — | — | *skipped: {la.skipped}* |")
    else:
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
        p_min = 1.0 / (int(p.get("n_perm", 1000)) + 1)
        p_min_str = f"< {p_min:.0e}" if oa.direction_p <= p_min else _pfmt(oa.direction_p)
        L.append(
            f"| Ordering × correctness | variant-permutation test on paired correctness "
            f"| χ² = {_fmt(oa.mcnemar_chi2, 1)} | {p_min_str} | **{oa.direction}** |"
        )
    if ca.skipped:
        L.append(f"| Calibration | — | — | — | *skipped: {ca.skipped}* |")
    else:
        L.append(
            f"| Calibration | ECE ({ca.binning}, {len(ca.bin_edges) - 1} bins) "
            f"| {_fmt(ca.ece)} [{_fmt(ca.ece_ci[0])}, {_fmt(ca.ece_ci[1])}] | — | **{ca.verdict}** |"
        )
        L.append(
            f"| Confidence discrimination | AUROC (confidence → correctness) "
            f"| {_fmt(ca.auroc)} [{_fmt(ca.auroc_ci[0])}, {_fmt(ca.auroc_ci[1])}] | — | — |"
        )
    if ab.skipped:
        L.append(f"| Selective prediction | — | — | — | *skipped: {ab.skipped}* |")
    else:
        L.append(
            f"| Selective prediction | AURC / E-AURC | {_fmt(ab.aurc)} / {_fmt(ab.e_aurc)} | — | — |"
        )
    L.append("")

    # ---- Position ----------------------------------------------------------
    L.append("## 1. Position / label bias")
    L.append("")
    if pa.skipped:
        L.append(f"*Skipped — {pa.skipped}*")
        L.append("")
    else:
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
            f"→ **{pa.marginal_verdict}**. First-position selection rate: {pa.first_rate:.3f} "
            f"(uniform would be {1.0 / k:.3f}). This marginal test assumes a balanced answer "
            "key (§2); with an imbalanced key, read the gold-offset test below instead."
        )
        L.append("")
        L.append("### Gold-offset test (confound-free)")
        L.append("")
        if pa.n_wrong == 0:
            L.append("*No wrong answers in the run — nothing to condition on (a good sign).*")
        else:
            L.append(
                "Among wrong answers, a position-blind model selects uniformly over the K−1 "
                "slots *relative to the gold one*: `(selected − gold) mod K` must be uniform "
                "over the non-zero offsets — independent of answer-key balance and accuracy. "
                f"n = {pa.n_wrong} wrong answers (one ordering per question):"
            )
            L.append("")
            L.append("| Offset (sel − gold) | " + " | ".join(f"+{d}" for d in range(1, k)) + " |")
            L.append("|---|" + "---|" * (k - 1))
            L.append("| Selection share | " + " | ".join(f"{a:.3f}" for a in pa.offset_rates) + " |")
            lo_row = " | ".join(f"{pa.offset_ci[d - 1, 0]:.3f}" for d in range(1, k))
            hi_row = " | ".join(f"{pa.offset_ci[d - 1, 1]:.3f}" for d in range(1, k))
            L.append(f"| 95% CI low | {lo_row} |")
            L.append(f"| 95% CI high | {hi_row} |")
            L.append("")
            L.append(
                f"Uniform would be {1.0 / (k - 1):.3f} per offset. "
                f"χ²({k - 2}) = {pa.offset_chi2:.1f}, p = {_pfmt(pa.offset_p)}, "
                f"w = {pa.offset_w:.3f} → **{pa.offset_verdict}**."
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
        L.append("| Accuracy | " + " | ".join(_fmt(a) for a in pa.accuracy_by_position) + " |")
        L.append("")
        L.append(f"Homogeneity χ² p = {_pfmt(pa.accuracy_p)}.")
        L.append("")

    # ---- Dataset side ------------------------------------------------------
    L.append("## 2. Answer-key balance (dataset side)")
    L.append("")
    L.append("| Gold at | " + " | ".join(letters) + " |")
    L.append("|---|" + "---|" * k)
    L.append("| Share of questions | " + " | ".join(_fmt(a) for a in pa.gold_distribution) + " |")
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
    if la.skipped:
        L.append(f"*Skipped — {la.skipped}*")
        L.append("")
    else:
        header = " | ".join(["shortest"] + [f"rank {i}" for i in range(1, k - 1)] + ["longest"]) if k > 2 else "shortest | longest"
        L.append("| Statistic | " + header + " |")
        L.append("|---|" + "---|" * k)
        L.append("| Selection rate | " + " | ".join(f"{a:.3f}" for a in la.selection_by_rank) + " |")
        L.append("| Gold share (dataset) | " + " | ".join(_fmt(a) for a in la.gold_rank_distribution) + " |")
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
            f"Directional asymmetry: McNemar χ² = {oa.mcnemar_chi2:.1f} on pooled pairs, "
            f"variant-permutation p = {p_min_str} → **{oa.direction}**."
        )
        if b.ordering_accuracy:
            L.append("")
            L.append("Accuracy by ordering (a spread here is the practical footprint of order sensitivity):")
            L.append("")
            L.append("| Ordering | n | Accuracy |")
            L.append("|---|---|---|")
            for name, n_v, acc in b.ordering_accuracy:
                L.append(f"| {name} | {n_v} | {acc:.3f} |")
    L.append("")

    # ---- Calibration -------------------------------------------------------
    L.append("## 5. Confidence calibration")
    L.append("")
    if ca.skipped:
        L.append(f"*Skipped — {ca.skipped}*")
    else:
        L.append(
            f"Accuracy = {ca.accuracy:.3f}, mean stated confidence = {ca.mean_confidence:.3f} "
            f"→ model is **{ca.direction}**. ECE = **{ca.ece:.3f}** "
            f"[{ca.ece_ci[0]:.3f}, {ca.ece_ci[1]:.3f}], MCE = {ca.mce:.3f} → **{ca.verdict}** "
            f"({ca.binning} binning, {len(ca.bin_edges) - 1} bins, n = {ca.n})."
        )
        L.append("")
        L.append(
            f"Discrimination: confidence AUROC = **{ca.auroc:.3f}** "
            f"[{ca.auroc_ci[0]:.3f}, {ca.auroc_ci[1]:.3f}] — probability that a random correct "
            "answer received higher confidence than a random wrong one. Calibration and "
            "discrimination are independent: a model can fail one and pass the other."
        )
        L.append("")
        L.append("![Reliability diagram](fig_reliability.png)")
    L.append("")

    # ---- Abstention --------------------------------------------------------
    L.append("## 6. Selective prediction (abstention)")
    L.append("")
    if ab.skipped:
        L.append(f"*Skipped — {ab.skipped}*")
    else:
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
    if pa.skipped is None and pa.verdict in ("minor", "moderate", "severe"):
        worst = int(np.nanargmax(pa.excess))
        recs.append(
            f"Position bias detected (largest marginal excess at **{letters[worst]}**, "
            f"{pa.excess[worst]:+.3f}). Report accuracy averaged over cyclic (or random) "
            "permutations of the options, or debias before comparing models."
        )
    if pa.skipped is None and pa.verdict == "inconclusive":
        recs.append(
            "Marginal slot preference detected, but the imbalanced answer key could explain "
            "it (the confound-free gold-offset test is silent). Re-balance the key — or "
            "re-run with permuted options — to attribute the signal."
        )
    if pa.skipped is None and pa.marginal_verdict not in ("none", "not available") and pa.gold_verdict != "none":
        recs.append(
            "The marginal position signal may partly reflect the imbalanced answer key "
            "(not only model behavior) — re-balance the key before drawing conclusions."
        )
    if pa.gold_verdict != "none":
        recs.append("Re-balance the answer key across positions before drawing model conclusions.")
    if la.skipped is None and la.verdict != "none":
        recs.append(
            "Length bias detected: control for option length (e.g. stratify accuracy by the "
            "gold option's length rank) or permute option order by length."
        )
    if la.gold_artifact == "artifact":
        recs.append(
            "Dataset artifact: the gold answer correlates with option length. Fix the benchmark "
            "or models will exploit the heuristic rather than the content."
        )
    if oa.skipped is None and oa.verdict == "unstable":
        recs.append(
            "Low ordering consistency: the reported score depends on which ordering you used. "
            "Average over orderings and report consistency alongside accuracy."
        )
    if oa.skipped is None and oa.direction == "systematic direction":
        recs.append(
            "Systematic correctness direction across orderings (often canonical-order "
            "memorization): treat single-order scores as optimistic."
        )
    if ca.skipped is None and ca.verdict in ("moderately miscalibrated", "poorly calibrated"):
        rec = (
            f"Calibration is {ca.direction}: consider temperature scaling on held-out data."
        )
        if ab.skipped is None and ab.coverage_at_target >= 0.05 and not np.isnan(ab.suggested_threshold):
            rec += (
                f" For risky downstream decisions, answer only when confidence ≥ "
                f"{ab.suggested_threshold:.2f} (risk ≤ {ab.target_risk:.0%})."
            )
        else:
            rec += " Confidence is too miscalibrated for thresholding — recalibrate first."
        recs.append(rec)
    skip_reasons = [a.skipped for a in (la, oa, ca, ab) if a.skipped]
    if skip_reasons:
        recs.append(
            "Enable richer logging to unlock skipped audits: record `option_lengths` "
            "(length audit), shuffled-order `variant_id`s plus `option_ids` (ordering audit), "
            "and the stated `confidence` (calibration and selective prediction)."
        )
    if not recs:
        recs.append("No significant bias detected at α = 0.01. Single-order scores remain "
                    "advisable to double-check on a fresh shuffle.")
    L.extend(f"- {r}" for r in recs)
    L.append("")

    L.append("## Method notes")
    L.append("")
    L.append(
        "- All CIs are cluster bootstrap percentile intervals (B = "
        f"{p.get('n_boot')} by default) with questions as clusters.\n"
        "- χ² and t-tests use one ordering per question: re-ordered variants of the same "
        "question are strongly correlated, and testing on all rows would overstate "
        "significance. This makes the test conservative on multi-variant runs; rates and "
        "CIs use every response.\n"
        "- χ² tests use α = 0.01; effect size is Cohen's w (0.1 / 0.2 / 0.3 ≈ small / medium / large). "
        "Multiple audits are reported per run, so treat borderline p-values with the "
        "family of tests in mind and lean on effect sizes and CIs.\n"
        "- The two position tests cover each other's confounds: the marginal test assumes a "
        "balanced answer key, while the gold-offset test (uniformity of `(selected − gold) "
        "mod K` among wrong answers) is immune to key imbalance and accuracy — but blind to "
        "absolute slot attraction when the key is balanced. Read them together; the combined "
        "verdict takes the more severe of the two.\n"
        "- The directional ordering p-value is a variant-label permutation test ("
        f"{p.get('n_perm')} permutations): under the null, variant labels are exchangeable "
        "within each question, which handles the correlated pairs that would break exact "
        "McNemar on multi-ordering runs.\n"
        "- E-AURC is the excess risk-coverage area over the oracle ordering (all correct "
        "answers first); smaller is better, 0 is unattainable in practice.\n"
        "- ECE is mildly upward-biased at small n (finite-sample noise inside bins); the "
        "reliability diagram and per-bin counts let you judge when bins are too sparse.\n"
        "- Abstain/unparsable answers are excluded from position, length and calibration "
        "statistics and counted in the abstain rate.\n"
        "- Equal-width bins can be noisy at the extremes; pass `--binning equal_mass` for "
        "quantile bins."
    )
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Pairwise (arena-style) reports
# ---------------------------------------------------------------------------

@dataclass
class PairwiseBundle:
    run_name: str
    n_judgments: int
    n_pairs: int
    pairwise: PairwiseAudit
    params: dict = field(default_factory=dict)


def run_pairwise_audit(
    run: PairwiseRun,
    n_boot: int = 1000,
    seed: int = 0,
) -> PairwiseBundle:
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    return PairwiseBundle(
        run_name=run.name,
        n_judgments=len(run),
        n_pairs=len({j.pair_id for j in run.judgments}),
        pairwise=audit_pairwise(run, n_boot=n_boot, seed=seed),
        params={
            "toolkit_version": __version__,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "n_boot": n_boot,
            "ci_alpha": 0.05,
            "test_alpha": 0.01,
            "seed": seed,
            "format": "pairwise",
        },
    )


def _write_pairwise_figures(bundle: PairwiseBundle, out: Path) -> None:
    pa = bundle.pairwise
    if pa.skipped is None:
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        labels = ["P(judge picks slot 0)", "P(content A in slot 0)"]
        values = [pa.pick_first_rate, pa.slot_balance_rate]
        yerr_lo = [pa.pick_first_ci[0] if not np.isnan(pa.pick_first_ci[0]) else values[0], values[1]]
        yerr_hi = [pa.pick_first_ci[1] if not np.isnan(pa.pick_first_ci[1]) else values[0], values[1]]
        ax.bar(labels, values, yerr=[np.array(values) - np.array(yerr_lo), np.array(yerr_hi) - np.array(values)],
               capsize=4, color=["#4C72B0", "#8C8C8C"], alpha=0.85)
        ax.axhline(0.5, color="crimson", ls="--", lw=1, label="balanced = 0.5")
        ax.set_ylim(0, max(0.75, float(np.nanmax(values)) * 1.3))
        ax.set_ylabel("rate")
        ax.set_title("Presentation-slot preference and balance")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "fig_pair_slot.png", dpi=150)
        plt.close(fig)

        top = pa.content_win_rates[:15]
        if top:
            fig, ax = plt.subplots(figsize=(6.0, 0.3 * len(top) + 1.2))
            names = [c.content_id for c in top][::-1]
            rates = np.array([c.win_rate for c in top][::-1])
            apps = np.array([c.appearances for c in top][::-1])
            se = np.sqrt(np.clip(rates * (1 - rates), 1e-6, None) / apps)
            err = 1.96 * se
            # keep the 95% CI inside [0, 1]: a win rate cannot leave that range
            lo = np.minimum(err, rates)
            hi = np.minimum(err, 1.0 - rates)
            ax.barh(names, rates, xerr=np.vstack([lo, hi]), capsize=2, color="#4C72B0", alpha=0.85)
            ax.axvline(0.5, color="crimson", ls="--", lw=1,
                       label="all-content mean = 0.5 (each match has one winner)")
            ax.set_xlabel("corrected win rate (95% CI)")
            ax.set_title(f"Top {len(top)} of {len(pa.content_win_rates)} contents by win rate\n"
                         "(head of the pool — above the 0.5 mean by selection)", fontsize=10)
            ax.set_xlim(0, 1)
            ax.legend(frameon=False, loc="lower right", fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "fig_pair_winrates.png", dpi=150)
            plt.close(fig)


def write_pairwise_report(bundle: PairwiseBundle, out_dir: str | Path, title: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_pairwise_figures(bundle, out)
    path = out / "report.md"
    path.write_text(_render_pairwise(bundle, title), encoding="utf-8")
    write_results_json(bundle, out / "results.json")
    return path


def _render_pairwise(b: PairwiseBundle, title: str | None) -> str:
    pa = b.pairwise
    p = b.params
    today = _dt.date.today().isoformat()
    title = title or f"Pairwise Preference Audit — {b.run_name}"

    L: list[str] = []
    L.append(f"# {title}")
    L.append("")
    L.append(
        f"*Generated {today} · {b.n_judgments} judgments · {b.n_pairs} pairs · "
        f"tie/undecided rate {pa.tie_rate:.1%}*"
    )
    L.append("")
    L.append(
        f"*Toolkit v{p.get('toolkit_version', '?')} · B = {p.get('n_boot')} bootstrap · "
        f"seed {p.get('seed')} · 95% CIs · tests at α = 0.01 — machine-readable results "
        "in `results.json`.*"
    )
    if b.n_pairs < 30:
        L.append("")
        L.append(
            f"**Small-sample caution:** only {b.n_pairs} pairs — CIs and tests are "
            "unstable at this size; treat every verdict as provisional."
        )
    L.append("")

    L.append("## Summary")
    L.append("")
    L.append("| Audit | Key statistic | Value | p-value | Verdict |")
    L.append("|---|---|---|---|---|")
    if pa.skipped:
        L.append(f"| Pairwise audit | — | — | — | *skipped: {pa.skipped}* |")
    else:
        L.append(
            f"| Slot preference | P(judge picks content shown first) "
            f"| {_fmt(pa.pick_first_rate)} [{_fmt(pa.pick_first_ci[0])}, {_fmt(pa.pick_first_ci[1])}] "
            f"| {_pfmt(pa.slot_p)} | **{pa.slot_verdict}** |"
        )
        L.append(
            f"| Slot balance (design side) | P(content A shown first) "
            f"| {_fmt(pa.slot_balance_rate)} | {_pfmt(pa.slot_balance_p)} | **{pa.slot_balance}** |"
        )
        if pa.n_swap_pairs:
            L.append(
                f"| Swap consistency | same winner across both presentation orders "
                f"| {_fmt(pa.swap_consistency)} [{_fmt(pa.swap_ci[0])}, {_fmt(pa.swap_ci[1])}] "
                f"| — | **{pa.swap_verdict}** |"
            )
        else:
            L.append("| Swap consistency | — | — | — | *skipped: judge each pair under both orders to enable* |")
        L.append(
            f"| Length preference | P(chosen is the longer description) "
            f"| {_fmt(pa.p_chosen_longer)} [{_fmt(pa.p_chosen_longer_ci[0])}, {_fmt(pa.p_chosen_longer_ci[1])}] "
            f"| — | **{pa.length_verdict}** |"
        )
    L.append("")

    L.append("## 1. Presentation-slot preference")
    L.append("")
    if pa.skipped:
        L.append(f"*Skipped — {pa.skipped}*")
    else:
        L.append(
            f"Across {pa.n_decided} decided judgments: the content shown **first** wins "
            f"{pa.pick_first_rate:.1%} of the time [{pa.pick_first_ci[0]:.3f}, {pa.pick_first_ci[1]:.3f}] "
            f"(binomial p = {_pfmt(pa.slot_p)} vs 0.5) → **{pa.slot_verdict}**."
        )
        L.append("")
        if pa.slot_balance == "imbalanced":
            L.append(
                "**Confound warning:** content A is presented first in "
                f"{pa.slot_balance_rate:.1%} of pairs (p = {_pfmt(pa.slot_balance_p)}) — the "
                "slot preference above is indistinguishable from a content-quality difference. "
                "Randomize which content sits in which slot and re-run."
            )
        else:
            L.append(
                f"Design check: content A sits in slot 0 for {pa.slot_balance_rate:.1%} of pairs "
                f"(p = {_pfmt(pa.slot_balance_p)}) — balanced, so the slot preference above is "
                "confound-free."
            )
        L.append("")
        L.append("![Slot preference](fig_pair_slot.png)")
    L.append("")

    L.append("## 2. Swap consistency")
    L.append("")
    if pa.skipped:
        L.append(f"*Skipped — {pa.skipped}*")
    elif pa.n_swap_pairs == 0:
        L.append("*Skipped — judge each pair under both presentation orders "
                 "(slot_of_a = 0 and 1) to measure order sensitivity.*")
    else:
        L.append(
            f"Across {pa.n_swap_pairs} swapped judgment pairs: the same content wins both times "
            f"**{pa.swap_consistency:.3f}** [{pa.swap_ci[0]:.3f}, {pa.swap_ci[1]:.3f}] → "
            f"**{pa.swap_verdict}**. Low consistency means the verdict depends on which "
            "order you happened to present — average over orders before drawing conclusions."
        )
    L.append("")

    L.append("## 3. Length preference")
    L.append("")
    if pa.skipped or np.isnan(pa.p_chosen_longer):
        L.append("*Skipped — needs context_lengths on the judgments.*")
    else:
        L.append(
            f"The chosen description is the longer one {pa.p_chosen_longer:.1%} of the time "
            f"[{pa.p_chosen_longer_ci[0]:.3f}, {pa.p_chosen_longer_ci[1]:.3f}] (coin flip 0.5) → "
            f"**{pa.length_verdict}**. Mean length advantage of the chosen side: "
            f"{pa.mean_length_advantage:+.1f} chars."
        )
    L.append("")

    L.append("## 4. Corrected leaderboard")
    L.append("")
    shown = pa.content_win_rates[:20]
    if not shown:
        L.append("*No decided judgments.*")
    else:
        L.append("| Content | Decided | Wins | Win rate | as first | as second |")
        L.append("|---|---|---|---|---|---|")
        for c in shown:
            L.append(
                f"| {c.content_id} | {c.appearances} | {c.wins} | {c.win_rate:.3f} "
                f"| {_fmt(c.win_rate_as_first)} | {_fmt(c.win_rate_as_second)} |"
            )
        if len(pa.content_win_rates) > len(shown):
            L.append(f"| … ({len(pa.content_win_rates) - len(shown)} more in results.json) | | | | | |")
        L.append("")
        L.append(
            "With a balanced design, the overall win rate is order-corrected; the as-first / "
            "as-second split exposes content that only wins from one slot."
        )
        L.append("")
        L.append("![Win rate by content](fig_pair_winrates.png)")
    L.append("")

    L.append("## Recommendations")
    L.append("")
    recs: list[str] = []
    if pa.skipped is None:
        if pa.slot_verdict in ("minor", "moderate", "severe") and pa.slot_balance == "balanced":
            recs.append(
                "Slot preference detected: randomize (or counterbalance) which content is "
                "presented first, and report order-corrected win rates."
            )
        elif pa.slot_verdict in ("minor", "moderate", "severe") and pa.slot_balance == "imbalanced":
            recs.append(
                "Slot preference signal is confounded by the imbalanced presentation design — "
                "randomize slot assignment before drawing conclusions."
            )
        if pa.swap_verdict == "unstable":
            recs.append(
                "Low swap consistency: judge verdicts depend on presentation order. Average "
                "over both orders, or tighten the judge prompt/rubric."
            )
        if pa.length_verdict in ("moderate", "severe"):
            recs.append(
                "Length preference detected: the judge favors longer descriptions — control "
                "for length (truncate/normalize descriptions) or report length-stratified rates."
            )
        if pa.tie_rate > 0.2:
            recs.append(
                f"Tie/undecided rate is high ({pa.tie_rate:.0%}): the judge often cannot decide. "
                "Consider sharper instructions or discard undecided pairs from rankings."
            )
    if not recs:
        recs.append("No significant judging bias detected at α = 0.01.")
    L.extend(f"- {r}" for r in recs)
    L.append("")

    L.append("## Method notes")
    L.append("")
    L.append(
        "- CIs are cluster bootstrap percentile intervals (B = "
        f"{p.get('n_boot')}), resampling pairs so repeated judgments of one pair move together.\n"
        "- Slot preference uses an exact binomial test vs 0.5 on decided judgments; it is "
        "confound-free only when the slot-assignment is balanced (checked in §1).\n"
        "- Swap consistency compares verdicts of the same pair under both presentation "
        "orders — order-sensitive judging shows up here, independent of slot balance.\n"
        "- Length preference measures P(the winner has the longer description) — a *combined* "
        "signal: judge preference for longer text AND any length–quality correlation in the "
        "content pool (strong contents often have longer descriptions). When it fires on a "
        "design you trust, inspect how descriptions were generated before blaming the judge.\n"
        "- Length CIs cluster on contents (contents recur across pairs), which widens them "
        "relative to per-judgment counting; the verdict thresholds account for this.\n"
        "- The corrected leaderboard splits each content's win rate by presentation slot; "
        "large first/second gaps indicate residual order effects.\n"
        "- Abstains/ties are excluded from preference statistics and reported as the tie rate."
    )
    return "\n".join(L) + "\n"
