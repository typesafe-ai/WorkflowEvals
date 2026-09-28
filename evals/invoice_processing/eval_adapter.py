from __future__ import annotations

from pathlib import Path

from core.contract import (
    Action,
    Case,
    CaseResult,
    Decision,
    Metric,
    Policy,
    Results,
    fingerprint,
)
from core.session import Session
from core.dataset import Dataset
from core.workflow import revision

from .structured import policy as domain
from .structured.decompose import run_case
from .structured.policy import PROFILES
from .structured.state import Packet

METRICS = [
    Metric(metric_id="exact_actions", title="Exact action-set agreement", comparison="exact_actions"),
    Metric(metric_id="primary_action", title="Primary action agreement", comparison="primary_action"),
]


def to_contract(decision: domain.Decision, revision: str | None) -> Decision:
    return Decision(
        policy_id=decision.policy,
        policy_revision=revision,
        actions=[Action(name=a.value) for a in decision.actions],
        primary_action=Action(name=decision.primary_action.value),
        outputs=decision.model_dump(mode="json", exclude={"policy", "actions", "primary_action", "trace"}),
    )


class Adapter:

    def __init__(self, root: Path, dataset: Dataset):
        self.revision = revision(root)
        self.implementation_revision = self.revision
        raw = dataset.inputs
        self.packets = {cid: Packet.model_validate(r) for cid, r in raw.items()}
        self.hashes = {cid: fingerprint(r) for cid, r in raw.items()}
        self.bundle = Results(
            eval_id="invoice_processing",
            title="Invoice processing",
            dataset_revision=fingerprint(self.hashes),
            cases=[Case(case_id=cid, content_hash=h) for cid, h in self.hashes.items()],
            policies=[
                Policy(policy_id=p, title=p.replace("_", " "), revision=self.revision) for p in PROFILES
            ],
            metrics=METRICS,
        )

    def execute(self, case_id: str, session: Session) -> CaseResult:
        return self.run_graph(case_id, session)

    def run_graph(self, case_id: str, session: Session) -> CaseResult:
        decomposition = run_case(
            self.packets[case_id],
            session,
            session.config.model,
            thinking=session.config.thinking,
            profiles=[PROFILES[p] for p in session.policy_ids] if session.policy_ids else None,
        )
        return CaseResult(
            case_id=case_id,
            decisions=[to_contract(d, self.revision) for d in decomposition.decisions.values()],
            details=decomposition.model_dump(mode="json"),
        )
