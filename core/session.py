from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Protocol

from pydantic import JsonValue
from core.question_types import SystemOneResponse, question_record, response_record

from .contract import Call, Cost, Execution, Observation, describe, fingerprint
from .pricing import estimated_cost
from .providers import ModelConfig


class QuestionClient(Protocol):
    def system_one(self, *, model, state, questions) -> SystemOneResponse: ...


def observations(node: str, state: JsonValue, questions: dict, response: dict) -> list[Observation]:
    out = []
    for qid, question in questions.items():
        q = question_record(question) if hasattr(question, "model_dump") else question
        a = response.get("answers", {}).get(qid)
        if a is None:
            continue
        kind = a["type"]
        if kind == "noul":
            p = a["noul"]
            probs = {"true": p, "false": 1 - p}
            value = p > 0.5 if p != 0.5 else None
        else:
            probs = a.get("probabilities") or {}
            # Classify scores by the modal level, not the rounded expected score.
            winners = [k for k, p in probs.items() if p == max(probs.values())] if probs else []
            value = winners[0] if len(winners) == 1 else None
        out.append(
            Observation(
                node_id=node,
                question_id=qid,
                kind=kind,
                value=value,
                probabilities=probs or None,
                confidence=max(probs.values()) if probs else None,
                confidence_method="max_probability" if probs else None,
                question_hash=fingerprint(q),
                state_hash=fingerprint(state),
                instructions=str(q.get("instructions") or ""),
            )
        )
    return out


GRAMMAR_LIMIT = "compiled grammar is too large"


def merge_responses(responses: list[SystemOneResponse]) -> SystemOneResponse:
    """Sum numeric usage fields; retain other metadata from the first response."""
    if len(responses) == 1:
        return responses[0]
    answers: dict = {}
    usage: dict = responses[0].usage.model_dump(mode="json", exclude_none=True)
    for response in responses:
        answers.update(response.answers)
    for response in responses[1:]:
        for key, value in response.usage.model_dump(mode="json", exclude_none=True).items():
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            if numeric and isinstance(usage.get(key), (int, float)):
                usage[key] = usage[key] + value
            else:
                usage.setdefault(key, value)
    return responses[0].model_copy(update={"answers": answers, "usage": type(responses[0].usage)(**usage)})


class Session:
    """Record one case; always use the configured model, ignoring the model argument to system_one."""

    def __init__(
        self,
        client: QuestionClient,
        model,
        config: ModelConfig,
        prices: dict,
        *,
        policy_ids: list[str] | None = None,
        question_workers: int = 8,
    ):
        self.client, self.model, self.config, self.prices = client, model, config, prices
        self.policy_ids = policy_ids
        self.question_workers = question_workers
        self.node = "semantic"
        self.calls: list[Call] = []
        self.observations: list[Observation] = []
        self._lock = threading.Lock()  # Per-question calls record concurrently.

    @contextmanager
    def _recorded(self, record: Call):
        with self._lock:
            record.call_id = str(len(self.calls) + 1)
            self.calls.append(record)
        start = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            record.status, record.error = "error", describe(exc)
            raise
        finally:
            record.wall_time_s = time.perf_counter() - start

    def _usage(self, record: Call, input_tokens, output_tokens, retries) -> None:
        record.input_tokens, record.output_tokens, record.retries = input_tokens, output_tokens, retries
        record.cost = estimated_cost(
            self.config.identity.name,
            input_tokens,
            output_tokens,
            self.prices,
        )

    def system_one(self, _model, state, questions):
        if self.config.question_mode == "one" and len(questions) > 1:
            with ThreadPoolExecutor(max_workers=self.question_workers) as pool:
                responses = list(pool.map(lambda item: self._ask(state, dict([item])), questions.items()))
            return merge_responses(responses)
        return self._ask_largest(state, questions)

    def _ask_largest(self, state, questions):
        """Split question sets only when the provider rejects the answer schema size."""
        record = self._call(state, questions)
        try:
            with self._recorded(record):
                return self._request(record, state, questions)
        except Exception as exc:
            if len(questions) < 2 or GRAMMAR_LIMIT not in describe(exc):
                raise
            with self._lock:  # The rejected schema produced no answers or billable usage.
                self.calls.remove(record)
        items = list(questions.items())
        halves = (dict(items[: len(items) // 2]), dict(items[len(items) // 2 :]))
        return merge_responses([self._ask_largest(state, half) for half in halves])

    def _ask(self, state, questions):
        record = self._call(state, questions)
        with self._recorded(record):
            return self._request(record, state, questions)

    def _call(self, state, questions) -> Call:
        wire = {k: question_record(q) for k, q in questions.items()}
        return Call(
            call_id="0",
            node_id=self.node,
            state=state,
            questions=wire,
            state_hash=fingerprint(state),
            questions_hash=fingerprint(wire),
            wall_time_s=0,
        )

    def _request(self, record: Call, state, questions):
        node, wire = self.node, record.questions
        response = self.client.system_one(model=self.model, state=state, questions=questions)
        raw = response_record(response)
        record.response = raw
        usage = raw.get("usage", {})
        self._usage(
            record,
            usage.get("input_tokens_total", usage.get("input_tokens")),
            usage.get("output_tokens_total", usage.get("output_tokens")),
            usage.get("n_retries"),
        )
        if set(response.answers) != set(questions):
            raise ValueError("Response question IDs differ from the requested question IDs")
        with self._lock:
            self.observations.extend(observations(node, state, wire, raw))
        return response


    def execution(self, policy_ids: list[str], elapsed: float, error: str | None) -> Execution:
        def total(field):
            values = [getattr(c, field) for c in self.calls]
            return sum(values) if all(v is not None for v in values) else None

        costs = [c.cost for c in self.calls]
        cost = None
        if all(c is not None for c in costs):
            cost = Cost(
                usd=sum(c.usd for c in costs),
                basis="estimated_uncached",
                pricing_revision=fingerprint(self.prices),
            )
        return Execution(
            policy_ids=policy_ids,
            policy_independent=True,
            status="error" if error else "ok",
            reason=error,
            cost=cost,
            input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"),
            wall_time_s=elapsed,
            summed_call_time_s=total("wall_time_s"),
            calls=len(self.calls),
        )
