from __future__ import annotations

import time
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from core.question_types import ChoiceAnswer, NoulAnswer, SystemOneResponse

from .facts import ExactFacts
from .questions import plan_questions
from .state import Packet


class SystemOneLike(Protocol):
    def system_one(self, model: str, state: Any, questions: Any) -> SystemOneResponse: ...


class Signals(BaseModel):
    """A missing probability means the question was not asked."""

    model_config = ConfigDict(extra="forbid")

    probabilities: dict[str, float | None]
    skipped: dict[str, str] = Field(default_factory=dict)
    model: str
    usage: dict[str, Any] = Field(default_factory=dict)

    def p(self, question_id: str, default: float | None = None) -> float | None:
        value = self.probabilities.get(question_id)
        return default if value is None else value

    def line(self, index: int, kind: str) -> float | None:
        return self.probabilities.get(f"line_{index}_{kind}")


def build_state(packet: Packet, facts: ExactFacts) -> dict[str, Any]:
    return {
        "task": (
            "Accounts-payable review packet. Evaluate each question about this invoice against "
            "the purchase order, contract, delivery evidence, prior invoices, vendor master "
            "record and communications. exact_facts were computed by code from the packet and "
            "are reliable; everything else is source evidence as received. The buyer's internal "
            "records (approver comments, tracker notes, receiving) outrank vendor claims."
        ),
        **packet.as_state(),
        "exact_facts": facts.model_dump(mode="json", exclude={"lines"}),
    }


def _to_signals(
    response: SystemOneResponse, asked: list[str], skipped: dict[str, str], elapsed: float | None = None
) -> Signals:
    probabilities: dict[str, float | None] = dict.fromkeys(skipped)
    for qid in asked:
        answer = response.answers[qid]
        if isinstance(answer, NoulAnswer):
            probabilities[qid] = answer.noul
        elif isinstance(answer, ChoiceAnswer):
            for option, prob in answer.probabilities.items():
                probabilities[f"{qid}_{option}"] = prob
        else:
            raise TypeError(f"question {qid!r}: unexpected answer type {answer.type}")
    usage = response.usage.model_dump(mode="json", exclude_none=True)
    if elapsed is not None:
        usage["wall_latency"] = round(elapsed, 3)
        usage.setdefault("latency", round(elapsed, 3))
    return Signals(probabilities=probabilities, skipped=skipped, model=response.model, usage=usage)


def ask_signals(client: SystemOneLike, model: str, packet: Packet, facts: ExactFacts) -> Signals:
    questions, skipped, _ = plan_questions(packet, facts)
    state = build_state(packet, facts)
    started = time.perf_counter()
    response = client.system_one(model, state, questions)
    return _to_signals(response, list(questions), skipped, time.perf_counter() - started)
