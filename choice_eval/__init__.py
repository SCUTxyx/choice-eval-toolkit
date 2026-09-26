"""choice-eval-toolkit: bias audit + calibration for multiple-choice evaluation runs.

Audits a set of MCQ responses (optionally under several option orderings) for:

* position / label bias (chi-square + effect size + cluster-bootstrap CIs),
* answer-key imbalance and gold-length artifacts on the dataset side,
* length bias (selection by length rank + within-question z-score),
* ordering consistency with a variant-permutation test for directional asymmetry,
* confidence calibration (ECE / MCE / reliability diagram / AUROC),
* selective prediction (risk-coverage, AURC, abstention threshold).

Everything runs on plain numpy/scipy — no model calls, no GPU, no datasets.
Missing inputs (confidence, option lengths, option ids) degrade to clearly
marked skipped sections instead of errors.
"""

from .abstention import AbstentionAudit, audit_abstention
from .arrays import RunArrays, to_arrays
from .calibration import CalibrationAudit, audit_calibration, confidence_auroc
from .generators import (
    expected_offset_rates,
    expected_pair_slot_rate,
    expected_p_chosen_longer,
    expected_selection_rates,
    expected_swap_consistency,
    generate_pairwise,
    generate_run,
)
from .length import LengthAudit, audit_length
from .order import OrderAudit, audit_order
from .pairwise import ContentWinRate, PairwiseAudit, audit_pairwise
from .position import PositionAudit, audit_position
from .report import (
    AuditBundle,
    PairwiseBundle,
    bundle_to_dict,
    run_audit,
    run_pairwise_audit,
    write_pairwise_report,
    write_report,
    write_results_json,
)
from .schema import (
    EvalRun,
    PairJudgment,
    PairwiseRun,
    Response,
    load_any,
    load_jsonl,
    load_pairwise_jsonl,
    save_jsonl,
    save_pairwise_jsonl,
)
from ._version import __version__

__all__ = [
    "EvalRun",
    "Response",
    "RunArrays",
    "AuditBundle",
    "PositionAudit",
    "LengthAudit",
    "OrderAudit",
    "CalibrationAudit",
    "AbstentionAudit",
    "generate_run",
    "expected_selection_rates",
    "to_arrays",
    "run_audit",
    "write_report",
    "write_results_json",
    "bundle_to_dict",
    "audit_position",
    "audit_length",
    "audit_order",
    "audit_calibration",
    "audit_abstention",
    "confidence_auroc",
    "load_jsonl",
    "save_jsonl",
    "__version__",
]
