from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1)]
Lab = Literal["openai", "anthropic", "typesafe", "other", "unknown"]


def fingerprint(value: JsonValue) -> str:
    wire = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(wire.encode()).hexdigest()


def describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModelIdentity(Record):
    name: Identifier
    lab: Lab = "unknown"
    effort: str | None = None


class Policy(Record):
    policy_id: Identifier
    title: str
    revision: str | None = None


class Metric(Record):
    metric_id: Identifier
    title: str
    comparison: Literal["exact_actions", "primary_action"]


class Case(Record):
    case_id: Identifier
    content_hash: Identifier
    tags: dict[str, str] = Field(default_factory=dict)


class Action(Record):
    name: Identifier
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    def key(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def actions(values: list[str | dict]) -> list[Action]:
    """Accept action names or {"action": name, **arguments}."""
    return [
        Action(name=a)
        if isinstance(a, str)
        else Action(name=a["action"], arguments={k: v for k, v in a.items() if k != "action"})
        for a in values
    ]


class Decision(Record):
    policy_id: Identifier
    policy_revision: str | None = None
    status: Literal["ok", "error", "missing", "unresolved"] = "ok"
    actions: list[Action] = Field(default_factory=list)
    primary_action: Action | None = None
    outputs: dict[str, JsonValue] = Field(default_factory=dict)
    reason: str | None = None

    @model_validator(mode="after")
    def valid_status(self) -> Self:
        if self.status != "ok" and (self.actions or self.primary_action is not None):
            raise ValueError("A non-ok decision cannot contain predicted actions")
        return self

    @classmethod
    def failed(cls, policy_id: str, policy_revision: str | None, reason: str) -> Decision:
        return cls(policy_id=policy_id, policy_revision=policy_revision, status="error", reason=reason)


class Cost(Record):
    usd: Nonnegative
    basis: Literal["estimated_uncached"]
    pricing_revision: str | None = None


class Execution(Record):
    """Usage for the requested policies, counting shared calls once."""

    policy_ids: list[Identifier]
    policy_independent: bool = False
    status: Literal["ok", "error", "partial"] = "ok"
    cost: Cost | None = None
    wall_time_s: Nonnegative | None = None
    summed_call_time_s: Nonnegative | None = None
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    calls: Annotated[int, Field(ge=0)] | None = None
    reason: str | None = None


class Observation(Record):

    node_id: Identifier
    question_id: Identifier
    kind: Literal["noul", "choice", "score", "finding", "unknown"]
    status: Literal["answered", "skipped"] = "answered"
    reason: str | None = None
    instructions: str | None = None
    value: JsonValue = None
    acceptable_values: list[JsonValue] | None = None
    probabilities: dict[str, Probability] | None = None
    confidence: Probability | None = None
    confidence_method: Literal["max_probability", "binary_margin", "sdk", "unknown"] | None = None
    question_hash: str | None = None
    state_hash: str | None = None
    definition_revision: str | None = None
    source: str | None = None
    origin: Literal["model_response", "dataset_annotation"] = "model_response"

    @model_validator(mode="after")
    def confidence_is_named(self) -> Self:
        if self.confidence is not None and self.confidence_method is None:
            raise ValueError("Confidence requires an explicit calculation method")
        return self


class Call(Record):
    """One logical call, including usage from retries when available."""

    call_id: Identifier
    kind: Literal["system_one"] = "system_one"
    node_id: Identifier
    state_hash: Identifier
    questions_hash: Identifier
    state: JsonValue
    questions: dict[str, JsonValue]
    instructions: str | None = None
    output_schema: dict[str, JsonValue] | None = None
    response: JsonValue = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    wall_time_s: Nonnegative
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    retries: Annotated[int, Field(ge=0)] | None = None
    cost: Cost | None = None

    @model_validator(mode="after")
    def hashes_match_payload(self) -> Self:
        if self.state_hash != fingerprint(self.state) or self.questions_hash != fingerprint(
            self.questions
        ):
            raise ValueError("Call fingerprints must match the recorded state and questions")
        return self


class CaseResult(Record):
    case_id: Identifier
    case_hash: str | None = None
    input_status: Literal["verified", "bound_annotation", "unverified", "stale"] = "unverified"
    decisions: list[Decision] = Field(default_factory=list)
    execution: Execution | None = None
    observations: list[Observation] = Field(default_factory=list)
    source: str | None = None
    calls: list[Call] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class Run(Record):
    run_id: Identifier
    model: ModelIdentity
    batching: str | None = None
    configuration_hash: str | None = None
    configuration: dict[str, JsonValue] = Field(default_factory=dict)
    repeat_group: str | None = None
    repeat_index: Annotated[int, Field(ge=1)] | None = None
    cases: list[CaseResult] = Field(default_factory=list)


class LabelSet(Record):
    label_set_id: Identifier
    title: str
    revision: Identifier
    method: Literal["independent", "consensus", "human", "dataset"]
    sources: list[ModelIdentity]
    source_run_ids: list[str] = Field(default_factory=list)
    cases: list[CaseResult] = Field(default_factory=list)


class Results(Record):
    schema_version: Literal["1.3"] = "1.3"
    eval_id: Identifier
    title: str
    dataset_revision: Identifier
    dataset_source: str | None = None
    policies: list[Policy]
    metrics: list[Metric] = Field(
        default_factory=lambda: [
            Metric(
                metric_id="exact_actions", title="Exact action-set agreement", comparison="exact_actions"
            )
        ]
    )
    cases: list[Case]
    references: dict[str, str] = Field(default_factory=dict)
    runs: list[Run] = Field(default_factory=list)
    label_sets: list[LabelSet] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


    @model_validator(mode="after")
    def identities_are_consistent(self) -> Self:
        def unique(values: list[str], what: str) -> set[str]:
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {what}")
            return set(values)

        case_ids = unique([c.case_id for c in self.cases], "case IDs")
        hashes = {c.case_id: c.content_hash for c in self.cases}
        policy_ids = unique([p.policy_id for p in self.policies], "policy IDs")
        unique([m.metric_id for m in self.metrics], "metric IDs")
        run_ids = unique([r.run_id for r in self.runs], "run IDs")
        label_ids = unique([s.label_set_id for s in self.label_sets], "label set IDs")
        if not set(self.references.values()) <= label_ids:
            raise ValueError("A reference points to an absent label set")
        for role, label_id in self.references.items():
            label = next(s for s in self.label_sets if s.label_set_id == label_id)
            if role == "consensus":
                if label.method != "consensus":
                    raise ValueError("The consensus reference must use consensus labels")
            elif role in ("openai", "anthropic") and (
                label.method != "independent" or {m.lab for m in label.sources} != {role}
            ):
                raise ValueError(f"The {role} reference must be an independent label set from that lab")
        for owner in [*self.runs, *self.label_sets]:
            unique([c.case_id for c in owner.cases], "result case IDs")
            if isinstance(owner, LabelSet) and not set(owner.source_run_ids) <= run_ids:
                raise ValueError("A label set references an absent source run")
            for result in owner.cases:
                if result.case_id not in case_ids:
                    raise ValueError(f"Unknown case {result.case_id}")
                if (
                    result.input_status in ("verified", "bound_annotation")
                    and result.case_hash != hashes[result.case_id]
                ):
                    raise ValueError(
                        "Verified results and bound annotations must match the dataset case hash"
                    )
                unique([c.call_id for c in result.calls], "call IDs")
                decisions = unique([d.policy_id for d in result.decisions], "decision policy IDs")
                if not decisions <= policy_ids:
                    raise ValueError("A decision references an unknown policy")
                if result.execution:
                    scope = unique(result.execution.policy_ids, "execution policy IDs")
                    if not scope or not scope <= policy_ids:
                        raise ValueError("Execution requires a nonempty, known policy scope")
                unique(
                    [f"{o.node_id}/{o.question_id}/{o.kind}" for o in result.observations], "observation IDs"
                )
        return self
