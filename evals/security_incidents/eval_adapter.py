from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from . import workflow
from core.answers import from_response

from core.contract import Action, Case, CaseResult, Decision, Metric, Policy, Results, fingerprint
from core.session import Session
from core.dataset import Dataset
from core.workflow import revision

POLICY = "playbook"


def tags(meta: dict[str, Any] | None) -> dict[str, str]:
    return {k: str(v) for k, v in (meta or {}).items() if isinstance(v, (str, int, float, bool))}


class Adapter:

    def __init__(self, root: Path, dataset: Dataset):
        self.states = dataset.inputs
        self.implementation_revision = revision(root)
        self.policy_revision = fingerprint(inspect.getsource(workflow))
        cases = [
            # Metadata is not model input, so exclude it from the input hash.
            Case(case_id=cid, content_hash=fingerprint(state), tags=tags(dataset.metadata[cid]))
            for cid, state in self.states.items()
        ]
        self.bundle = Results(
            eval_id="security_incidents",
            title="Security incidents",
            cases=cases,
            dataset_revision=fingerprint({c.case_id: c.content_hash for c in cases}),
            policies=[Policy(policy_id=POLICY, title="Alert-triage playbook", revision=self.policy_revision)],
            metrics=[
                Metric(metric_id="exact_actions", title="Exact action-set agreement", comparison="exact_actions")
            ],
        )

    def execute(self, case_id: str, session: Session) -> CaseResult:
        state = self.states[case_id]
        return self.run_graph(case_id, state, session)

    def run_graph(self, case_id: str, state: dict[str, Any], session: Session) -> CaseResult:
        def judge(state, node_id, questions):
            session.node = node_id
            return from_response(session.system_one(session.config.model, state, questions))

        trace = workflow.run(state, judge)
        return CaseResult(
            case_id=case_id,
            decisions=[self.decision(trace.label, band=trace.band, state=trace.to_json()["state"])],
            details=trace.to_json(),
        )


    def decision(self, label: str, **outputs: Any) -> Decision:
        action = Action(name=label)
        return Decision(
            policy_id=POLICY,
            policy_revision=self.policy_revision,
            actions=[action],
            primary_action=action,
            outputs=outputs,
        )
