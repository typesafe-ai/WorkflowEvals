from __future__ import annotations

from pathlib import Path

from .policies import POLICIES
from .answers import from_system_one_response
from .state import Case as DomainCase
from .workflow import merged_to_json, run_workflow
from core.contract import (
    Case,
    CaseResult,
    Decision,
    Metric,
    Policy,
    Results,
    actions,
    fingerprint,
)
from core.session import Session
from core.dataset import Dataset
from core.workflow import revision


class Adapter:

    def __init__(self, root: Path, dataset: Dataset):
        self.cases = {
            cid: DomainCase.from_json({**dataset.metadata[cid], **record})
            for cid, record in dataset.inputs.items()
        }
        self.revision = revision(root)
        self.implementation_revision = self.revision
        cases = [Case(case_id=cid, content_hash=fingerprint(r)) for cid, r in dataset.inputs.items()]
        self.bundle = Results(
            eval_id="customer_service",
            title="Customer service",
            cases=cases,
            dataset_revision=fingerprint({c.case_id: c.content_hash for c in cases}),
            policies=[Policy(policy_id=p, title=p, revision=self.revision) for p in POLICIES],
            metrics=[
                Metric(metric_id="exact_actions", title="Exact action-set agreement", comparison="exact_actions")
            ],
        )

    def execute(self, case_id: str, session: Session) -> CaseResult:
        return self.run_graph(case_id, session)

    def run_graph(self, case_id: str, session: Session) -> CaseResult:
        def ask(node, state, questions):
            session.node = node
            response = session.system_one(session.config.model, state, questions)
            return from_system_one_response(response, questions)

        case = self.cases[case_id]
        answers, trace = run_workflow(ask, case)
        return CaseResult(
            case_id=case_id,
            decisions=[
                Decision(
                    policy_id=p, policy_revision=self.revision, actions=actions(pol.decide(answers, case))
                )
                for p, pol in POLICIES.items()
                if p in session.policy_ids
            ],
            details={"answers": merged_to_json(answers), "trace": [n.to_json() for n in trace]},
        )
