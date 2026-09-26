"""choice-eval-toolkit: bias audit + calibration for multiple-choice evaluation runs.

Audits a set of MCQ responses (optionally under several option orderings) for:

* position / label bias (chi-square + effect size + bootstrap CIs),
* answer-key imbalance and gold-length artifacts on the dataset side,
* length bias (selection by length rank + within-question z-score),
* ordering consistency with McNemar's test on paired correctness,
* confidence calibration (ECE / MCE / reliability diagram),
* selective prediction (risk-coverage, AURC, abstention threshold).

Everything runs on plain numpy/scipy — no model calls, no GPU, no datasets.
"""

from .abstention import AbstentionAudit, audit_abstention
from .arrays import RunArrays, to_arrays
from .calibration import CalibrationAudit, audit_calibration
from .generators import generate_run
from .length import LengthAudit, audit_length
from .order import OrderAudit, audit_order
from .position import PositionAudit, audit_position
from .report import AuditBundle, run_audit, write_report
from .schema import EvalRun, Response, load_jsonl, save_jsonl

__version__ = "0.1.0"

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
    "to_arrays",
    "run_audit",
    "write_report",
    "audit_position",
    "audit_length",
    "audit_order",
    "audit_calibration",
    "audit_abstention",
    "load_jsonl",
    "save_jsonl",
    "__version__",
]
