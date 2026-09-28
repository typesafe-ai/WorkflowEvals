"""Node gates select which questions run; policy thresholds decide the actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .questions import CONSENT, FRAUD, MONEY, RETENTION, TRIAGE, integrity_questions, intent_pair
from .state import Case
from core.answers import gap, ranked
from core.question_types import Answers, ChoiceAnswer, NoulAnswer, QuestionSet
from .answers import from_json as answer_from_json, to_json as answer_to_json

FRAUD_FLOOR = 0.30
MONEY_FLOOR = 0.20
RETENTION_FLOOR = 0.15
PAIR_GAP = 0.25
MONEY_INTENTS = ("refund_request", "billing_dispute")

Asker = Callable[[str, Dict[str, Any], QuestionSet], Answers]


@dataclass
class NodeRun:
    node: str
    ran: bool
    reason: str
    state: Optional[Dict[str, Any]] = None
    n_questions: int = 0
    answers: Optional[Dict[str, Any]] = None
    calls: int = 0

    def to_json(self) -> dict:
        return {"node": self.node, "ran": self.ran, "reason": self.reason, "n_questions": self.n_questions,
                "answers": self.answers, "calls": self.calls, "state": self.state}


def run_workflow(ask: Asker, case: Case) -> Tuple[Answers, List[NodeRun]]:
    trace: List[NodeRun] = []
    merged: Answers = {}

    def run(node: str, state: Dict[str, Any], qs: QuestionSet, reason: str) -> Answers:
        ans = ask(node, state, qs)
        merged.update(ans)
        merged[f"__ran__{node}"] = NoulAnswer(noul=1.0)
        trace.append(NodeRun(node, True, reason, state, len(qs), {k: answer_to_json(a) for k, a in ans.items()}, 1))
        return ans

    def skip(node: str, reason: str) -> None:
        trace.append(NodeRun(node, False, reason))

    tri = run("triage", case.triage_state(), TRIAGE, "always")
    intent = tri["intent"]
    p_unauth = tri["reports_unauthorized"].noul
    p_money = tri["desired_outcome"].probabilities["money_back"]
    p_cancel = intent.probabilities["cancel_account"]

    if case.state.pending:
        run("consent", case.consent_state(), CONSENT, f"proposal `{case.state.pending}` is pending")
    else:
        skip("consent", "no proposal is pending")

    if p_unauth >= FRAUD_FLOOR or intent.choice == "unauthorized_charge":
        run("fraud", case.fraud_state(), FRAUD, f"reports_unauthorized {p_unauth:.2f} ≥ {FRAUD_FLOOR} or intent leads with unauthorized_charge")
    else:
        skip("fraud", f"reports_unauthorized {p_unauth:.2f} < {FRAUD_FLOOR} and intent does not lead with unauthorized_charge")

    money_on_file = case.account.refund is not None or case.account.outstanding_fee_usd > 0
    if money_on_file and (p_money >= MONEY_FLOOR or intent.choice in MONEY_INTENTS):
        run("money", case.money_state(), MONEY, f"money on file and (P(money_back) {p_money:.2f} ≥ {MONEY_FLOOR} or a money intent leads)")
    elif not money_on_file:
        skip("money", "no refund claim or fee on file")
    else:
        skip("money", f"P(money_back) {p_money:.2f} < {MONEY_FLOOR} and no money intent leads")

    if intent.choice == "cancel_account" or (case.account.subscription_active and p_cancel >= RETENTION_FLOOR):
        run("retention", case.retention_state(), RETENTION, f"cancel leads or P(cancel) {p_cancel:.2f} ≥ {RETENTION_FLOOR} with an active subscription")
    else:
        skip("retention", f"cancel does not lead and P(cancel) {p_cancel:.2f} < {RETENTION_FLOOR} (or no subscription)")

    if gap(intent) < PAIR_GAP:
        a, b = [k for k, _ in ranked(intent)[:2]]
        pair = run("intent_pair", case.customer_turns_state(), intent_pair(a, b), f"triage intent gap {gap(intent):.2f} < {PAIR_GAP}: {a} vs {b}")["intent_pair"]
        merged["intent_triage"] = intent
        mass = intent.probabilities[a] + intent.probabilities[b]
        probs = dict(intent.probabilities)
        probs[a], probs[b] = mass * pair.probabilities[a], mass * pair.probabilities[b]
        n = len(probs); top = max(probs.values())
        merged["intent"] = ChoiceAnswer(choice=max(probs, key=probs.__getitem__), probabilities=probs, confidence=(top - 1 / n) / (1 - 1 / n))
    else:
        skip("intent_pair", f"triage intent gap {gap(intent):.2f} ≥ {PAIR_GAP}")

    new_msgs = case.new_assistant_message_numbers()
    ran_fraud, ran_money = "__ran__fraud" in merged, "__ran__money" in merged
    if new_msgs and (ran_fraud or ran_money):
        ig = run("integrity", case.integrity_state(), integrity_questions(new_msgs), f"{len(new_msgs)} new assistant message(s) and a {'fraud' if ran_fraud else 'money'} node ran")
        for claim, prefix in (("claims_refund_done", "refund_done_msg_"), ("claims_secured", "secured_msg_")):
            ps = {k: ig[f"{prefix}{k}"].noul for k in new_msgs}
            p_any = 1.0
            for p in ps.values():
                p_any *= 1.0 - p
            merged[claim] = NoulAnswer(noul=1.0 - p_any)
            merged[f"{claim}_msg"] = NoulAnswer(noul=float(max(ps, key=ps.get)))
    elif not new_msgs:
        skip("integrity", "no new assistant message to check")
    else:
        skip("integrity", "neither the fraud nor the money node ran: nothing the record could contradict")

    return merged, trace


def merged_to_json(merged: Answers) -> Dict[str, Any]:
    return {k: answer_to_json(a) for k, a in merged.items()}


def merged_from_json(d: Dict[str, Any]) -> Answers:
    return {k: answer_from_json(v) for k, v in d.items()}
