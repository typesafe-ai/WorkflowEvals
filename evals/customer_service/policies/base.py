"""Apply ordered rules; keep the first response action and deduplicate other actions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .. import actions as A
from ..gates import Cond, forms_used
from core.question_types import Answers
from ..state import Case

Emitter = Callable[[Answers, Case], Optional[A.Action]]


@dataclass(frozen=True)
class Emit:
    text: str
    fn: Emitter = field(compare=False, hash=False)


@dataclass(frozen=True)
class Rule:
    name: str
    when: Cond
    then: Sequence[Emit]
    stop: bool = True
    why: str = ""

    def actions(self, ans: Answers, case: Case) -> List[A.Action]:
        out = []
        for e in self.then:
            a = e.fn(ans, case)
            if a is not None:
                out.append(a)
        return out


@dataclass(frozen=True)
class Section:
    title: str
    rules: Sequence[Rule]
    mode: str = "first"
    intro: str = ""


@dataclass
class Policy:
    key: str
    name: str
    summary: str
    sections: Sequence[Section]

    def sections_for(self, ans: Answers, case: Case) -> Sequence[Section]:
        return self.sections

    def decide(self, ans: Answers, case: Case) -> List[A.Action]:
        if case.state.handed_off:
            return []
        out: List[A.Action] = []
        for sec in self.sections_for(ans, case):
            stop = False
            for r in sec.rules:
                if r.when.holds(ans, case):
                    out.extend(r.actions(ans, case))
                    stop = stop or r.stop
                    if sec.mode == "first":
                        break
            if stop:
                break
        return finalize(out)

    def forms(self) -> set:
        s = set()
        for sec in self.sections:
            for r in sec.rules:
                s |= forms_used(r.when)
        return s


def finalize(actions: List[A.Action]) -> List[A.Action]:
    seen, out, responded = set(), [], False
    for a in actions:
        k = A.canonical([a])[0]
        if a["action"] == "respond_with":
            if responded:
                continue
            responded = True
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out


def R(message: str) -> Emit:
    return Emit(f"respond_with `{message}`", lambda ans, c, m=message: A.respond_with(m))


def do(fn: Callable[[], A.Action], text: str) -> Emit:
    return Emit(text, lambda ans, c: fn())


FREEZE = Emit("`freeze_card` (the card on file)", lambda ans, c: A.freeze_card(c.account.card_id))
REFUND = Emit("`issue_refund` (the amount on file)", lambda ans, c: A.issue_refund(c.account.refund.amount_usd))
WAIVE_FEE = Emit("`issue_refund` (the outstanding fee)", lambda ans, c: A.issue_refund(c.account.outstanding_fee_usd))
DISPUTE = Emit("`open_dispute` (the transaction on file)", lambda ans, c: A.open_dispute(c.account.transaction.transaction_id))
SET_INTENT = Emit("`set_intent` (the top intent)", lambda ans, c: A.set_intent(ans["intent"].choice))
CLEAR_PENDING = Emit("`clear_pending`", lambda ans, c: A.clear_pending())
MARK_RESOLVED = Emit("`mark_resolved`", lambda ans, c: A.mark_resolved())
CLOSE = Emit("`close_ticket`", lambda ans, c: A.close_ticket())
REDACT = Emit("`redact_transcript`", lambda ans, c: A.redact_transcript())
RETENTION = Emit("`apply_retention_offer`", lambda ans, c: A.apply_retention_offer())
CANCEL = Emit("`cancel_subscription`", lambda ans, c: A.cancel_subscription())


def pending(p: str) -> Emit:
    return Emit(f"`set_pending` `{p}`", lambda ans, c, p=p: A.set_pending(p))


def handoff(q: str) -> Emit:
    return Emit(f"`handoff_human` → `{q}`", lambda ans, c, q=q: A.handoff_human(q))


def flag(reason: str) -> Emit:
    return Emit(f"`flag_for_review` `{reason}`", lambda ans, c, r=reason: A.flag_for_review(r))


def priority(level: str) -> Emit:
    return Emit(f"`set_priority` `{level}`", lambda ans, c, l=level: A.set_priority(l))
