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
from core.answers import from_response
from .gates import PROFILES
from .trace import Run, parse
from .workflow import run_workflow

METRICS = [
    # One action per run makes exact-action agreement redundant.
    Metric(metric_id="primary_action", title="Primary action agreement", comparison="primary_action"),
]


class Adapter:

    def __init__(self, root: Path, dataset: Dataset):
        self.raw = dataset.inputs
        self.runs: dict[str, Run] = {tid: parse(record) for tid, record in self.raw.items()}
        self.revision = revision(root)
        self.implementation_revision = self.revision
        cases = [Case(case_id=tid, content_hash=fingerprint(record)) for tid, record in self.raw.items()]
        self.bundle = Results(
            eval_id="agent_trace_observability",
            title="Agent trace triage",
            cases=cases,
            dataset_revision=fingerprint({c.case_id: c.content_hash for c in cases}),
            policies=[Policy(policy_id=p, title=f"{p} profile", revision=self.revision) for p in PROFILES],
            metrics=METRICS,
        )

    def execute(self, case_id: str, session: Session) -> CaseResult:
        return self.run_graph(case_id, session)


    def run_graph(self, case_id: str, session: Session) -> CaseResult:
        def ask(node, state, questions):
            session.node = node
            return from_response(session.system_one(session.config.model, state, questions))

        result = run_workflow(ask, self.runs[case_id])
        return CaseResult(
            case_id=case_id,
            decisions=[
                Decision(
                    policy_id=name,
                    policy_revision=self.revision,
                    actions=[Action(name=v.action.value)],
                    primary_action=Action(name=v.action.value),
                    outputs=v.to_outputs(),
                )
                for name, v in result.verdicts.items()
                if name in session.policy_ids
            ],
            details=result.to_details(),
        )
