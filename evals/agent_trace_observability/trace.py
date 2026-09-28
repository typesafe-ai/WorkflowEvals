"""Parse exported spans into conversation, tool calls, and feedback kept separately."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

INTENT_SPAN = "intent_classification"
FEEDBACK_SPAN = "user_feedback"

POLICY_HEADER = "Your policy:"
ENTITLEMENT_HEADER = "What your role is entitled to:"
AUTONOMY_HEADER = "How much you may do on your own:"
HANDOFF_HEADER = "When you hand off, it goes to:"
ENTITLEMENT_LINE = re.compile(r"^-\s*(?P<operation>[^:]+?):\s*(?P<text>.+)$")

CONTEXT_KEYS = (
    "env",
    "deployment_id",
    "agent_version",
    "policy_version",
    "channel",
    "locale",
    "customer_segment",
    "authenticated",
)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Turn:
    """Conversational text only; tool results belong to calls."""

    role: str
    text: str

    @property
    def is_user(self) -> bool:
        return self.role == "user"


@dataclass(frozen=True)
class ToolSpec:

    name: str  # Service.operation
    description: str
    read_only: bool
    destructive: bool
    idempotent: bool


@dataclass(frozen=True)
class Grant:
    """Keep the entitlement text even when its wording does not match the flag parser."""

    operation: str
    text: str
    entitled: bool
    requires_confirmation: bool
    supervisor_approval: bool

    @property
    def conditioned(self) -> bool:
        return not self.entitled or self.requires_confirmation or self.supervisor_approval


@dataclass(frozen=True)
class Charter:

    agent_name: str
    text: str
    purpose: str
    rules: tuple[str, ...]
    entitlements: dict[str, Grant]
    autonomy: str
    handoff_to: str

    def grant_for(self, operation: str) -> Grant | None:
        return self.entitlements.get(operation)


@dataclass(frozen=True)
class ToolCall:

    ordinal: int  # 1-based step number
    name: str  # Service.operation, or unknown.unparsed
    arguments: dict[str, Any]
    result: str | None
    error_message: str | None
    error_code: str | None
    duration_s: float | None
    turn_index: int = 0  # Number of conversational turns before this call.

    @property
    def failed(self) -> bool:
        return self.error_code is not None or self.error_message is not None

    @property
    def service(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def label(self) -> str:
        return f"step {self.ordinal}: {self.name}"


@dataclass(frozen=True)
class Reaction:

    signal: str
    csat: int | None
    comment: str
    trailing_message: str
    reopened: bool

    @property
    def is_silent(self) -> bool:
        return not (self.comment or self.trailing_message)


@dataclass(frozen=True)
class Guardrail:
    name: str
    triggered: bool


@dataclass(frozen=True)
class Handoff:
    from_agent: str
    to_agent: str


@dataclass(frozen=True)
class Run:

    trace_id: str
    workflow_name: str
    group_id: str
    started_at: datetime | None
    ended_at: datetime | None
    charter: Charter
    tools: dict[str, ToolSpec]
    context: dict[str, Any]  # Customer framing, not a permission grant.
    turns: list[Turn]
    calls: list[ToolCall]
    reaction: Reaction | None
    guardrails: list[Guardrail]
    handoffs: list[Handoff]
    intent_category: str | None  # Platform classification: action or research.
    intent_confidence: float | None
    input_tokens: int
    output_tokens: int
    generations: int
    raw_metadata: dict[str, Any] = field(default_factory=dict, repr=False)


    @property
    def opening_request(self) -> str:
        return next((t.text for t in self.turns if t.is_user), "")

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.is_user]

    @property
    def final_message(self) -> str:
        return next((t.text for t in reversed(self.turns) if not t.is_user), "")

    @property
    def failed_calls(self) -> list[ToolCall]:
        return [c for c in self.calls if c.failed]

    def spec_for(self, call: ToolCall) -> ToolSpec | None:
        return self.tools.get(call.name)

    def grant_for(self, call: ToolCall) -> Grant | None:
        return self.charter.grant_for(call.name)

    def is_consequential(self, call: ToolCall) -> bool:
        """Destructive calls or calls conditioned on confirmation, approval, or entitlement."""
        spec, grant = self.spec_for(call), self.grant_for(call)
        return bool(spec and spec.destructive) or bool(grant and grant.conditioned)

    @property
    def consequential_calls(self) -> list[ToolCall]:
        return [c for c in self.calls if self.is_consequential(c)]

    @property
    def destructive_calls(self) -> list[ToolCall]:
        return [c for c in self.calls if (s := self.spec_for(c)) and s.destructive]

    def turns_before(self, call: ToolCall) -> list[Turn]:
        return self.turns[: call.turn_index]

    @property
    def duration_s(self) -> float | None:
        if not (self.started_at and self.ended_at):
            return None
        return (self.ended_at - self.started_at).total_seconds()


def _ordered_spans(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(raw.get("spans", []), key=lambda s: (s.get("started_at") or "", s.get("id") or ""))


def _generations(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s["span_data"] for s in spans if s.get("span_data", {}).get("type") == "generation"]


def _message_key(message: dict[str, Any]) -> str:
    return json.dumps(
        [message.get("role"), message.get("content"), message.get("tool_call_id"), message.get("tool_calls")],
        sort_keys=True,
        default=str,
    )


def _conversation(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge repeated histories and incremental messages in span order; exclude system messages."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for data in _generations(spans):
        for message in list(data.get("input") or []) + list(data.get("output") or []):
            if message.get("role") == "system":
                continue
            key = _message_key(message)
            if key in seen:
                continue
            seen.add(key)
            out.append(message)
    return out


def _system_prompt(spans: list[dict[str, Any]]) -> str:
    for data in _generations(spans):
        for message in data.get("input") or []:
            if message.get("role") == "system" and message.get("content"):
                return str(message["content"])
    return ""


def _section(text: str, header: str) -> str:
    if header not in text:
        return ""
    body = text.split(header, 1)[1]
    return body.split("\n\n", 1)[0].strip()


def parse_charter(text: str, agent_name: str) -> Charter:
    """Preserve the whole prompt when policy or entitlement sections cannot be recognized."""
    purpose = text.strip().split("\n\n", 1)[0].strip() if text.strip() else ""
    rules = tuple(
        line.strip()[1:].strip()
        for line in _section(text, POLICY_HEADER).splitlines()
        if line.strip().startswith("-")
    )
    entitlements: dict[str, Grant] = {}
    for line in _section(text, ENTITLEMENT_HEADER).splitlines():
        match = ENTITLEMENT_LINE.match(line.strip())
        if not match:
            continue
        operation, body = match.group("operation").strip(), match.group("text").strip()
        lowered = body.lower()
        grant = Grant(
            operation=operation,
            text=body,
            entitled="not entitled" not in lowered,
            requires_confirmation="explicit confirmation" in lowered,
            supervisor_approval="supervisor" in lowered,
        )
        # Repeated entitlement lines use the stricter interpretation.
        earlier = entitlements.get(operation)
        if earlier is None or (grant.conditioned and not earlier.conditioned):
            entitlements[operation] = grant
    return Charter(
        agent_name=agent_name,
        text=text,
        purpose=purpose,
        rules=rules,
        entitlements=entitlements,
        autonomy=_section(text, AUTONOMY_HEADER),
        handoff_to=_section(text, HANDOFF_HEADER),
    )


def _tool_definitions(spans: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Accept tool definitions on the agent span or repeated generation spans."""
    for span in spans:
        data = span.get("span_data", {})
        if data.get("type") in ("agent", "generation"):
            for definition in data.get("tools") or []:
                if isinstance(definition, dict) and isinstance(definition.get("function"), dict):
                    yield definition


def _tool_catalog(spans: list[dict[str, Any]]) -> dict[str, ToolSpec]:
    catalog: dict[str, ToolSpec] = {}
    for definition in _tool_definitions(spans):
        function = definition["function"]
        name = function.get("name")
        if not name or name in catalog:
            continue
        hints = definition.get("annotations") or {}
        catalog[name] = ToolSpec(
            name=name,
            description=str(function.get("description") or ""),
            read_only=bool(hints.get("readOnlyHint", False)),
            destructive=bool(hints.get("destructiveHint", False)),
            idempotent=bool(hints.get("idempotentHint", False)),
        )
    return catalog


def _arguments(blob: Any) -> dict[str, Any]:
    if isinstance(blob, dict):
        return blob
    if isinstance(blob, str) and blob.strip():
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            return {"_unparsed": blob}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {}


def _is_turn(message: dict[str, Any]) -> bool:
    return message.get("role") in ("user", "assistant") and bool((message.get("content") or "").strip())


def _tool_calls(spans: list[dict[str, Any]], messages: list[dict[str, Any]]) -> list[ToolCall]:
    """Join conversation calls to function spans by order within each tool name."""
    results: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id"):
            results[message["tool_call_id"]] = message.get("content") or ""

    spans_by_tool: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        data = span.get("span_data", {})
        if data.get("type") == "function":
            spans_by_tool.setdefault(str(data.get("name") or ""), []).append(span)

    calls: list[ToolCall] = []
    used: dict[str, int] = {}
    turns_seen = 0
    for message in messages:
        if _is_turn(message):
            turns_seen += 1
        for raw_call in message.get("tool_calls") or []:
            function = raw_call.get("function", {})
            name = str(function.get("name") or "")
            index = used.get(name, 0)
            used[name] = index + 1
            candidates = spans_by_tool.get(name, [])
            span = candidates[index] if index < len(candidates) else None
            error = (span or {}).get("error") or {}
            started = _parse_time((span or {}).get("started_at"))
            ended = _parse_time((span or {}).get("ended_at"))
            # Fall back to span output when the run ended before recording a tool message.
            result = results.get(raw_call.get("id") or "")
            if result is None and span is not None:
                span_output = span.get("span_data", {}).get("output")
                result = span_output if isinstance(span_output, str) else None
            calls.append(
                ToolCall(
                    ordinal=len(calls) + 1,
                    name=name,
                    arguments=_arguments(function.get("arguments")),
                    result=result,
                    error_message=error.get("message"),
                    error_code=(error.get("data") or {}).get("code"),
                    duration_s=(ended - started).total_seconds() if started and ended else None,
                    turn_index=turns_seen,
                )
            )
    return calls


def _custom(spans: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for span in spans:
        data = span.get("span_data", {})
        if data.get("type") == "custom" and data.get("name") == name:
            payload = data.get("data")
            return payload if isinstance(payload, dict) else None
    return None


def _agent_name(spans: list[dict[str, Any]], raw: dict[str, Any]) -> str:
    for span in spans:
        data = span.get("span_data", {})
        if data.get("type") == "agent" and data.get("name"):
            return str(data["name"])
    return str(raw.get("workflow_name", ""))


def parse(raw: dict[str, Any]) -> Run:
    spans = _ordered_spans(raw)
    metadata = raw.get("metadata", {})
    messages = _conversation(spans)
    turns = [Turn(role=m["role"], text=(m.get("content") or "").strip()) for m in messages if _is_turn(m)]

    feedback = _custom(spans, FEEDBACK_SPAN)
    reaction = None
    if feedback is not None:
        csat = feedback.get("csat")
        reaction = Reaction(
            signal=str(feedback.get("signal") or "none"),
            csat=int(csat) if isinstance(csat, (int, float)) else None,
            comment=(feedback.get("comment") or "").strip(),
            trailing_message=(feedback.get("trailing_message") or "").strip(),
            reopened=bool(feedback.get("reopened_within_24h")),
        )

    intent = _custom(spans, INTENT_SPAN) or {}
    usage = [g.get("usage") or {} for g in _generations(spans)]
    system_prompt = _system_prompt(spans)

    return Run(
        trace_id=raw.get("id", ""),
        workflow_name=raw.get("workflow_name", ""),
        group_id=raw.get("group_id", ""),
        started_at=_parse_time(raw.get("started_at")),
        ended_at=_parse_time(raw.get("ended_at")),
        charter=parse_charter(system_prompt, _agent_name(spans, raw)),
        tools=_tool_catalog(spans),
        context={k: metadata[k] for k in CONTEXT_KEYS if k in metadata},
        turns=turns,
        calls=_tool_calls(spans, messages),
        reaction=reaction,
        guardrails=[
            Guardrail(name=s["span_data"].get("name", ""), triggered=bool(s["span_data"].get("triggered")))
            for s in spans
            if s["span_data"].get("type") == "guardrail"
        ],
        handoffs=[
            Handoff(
                from_agent=s["span_data"].get("from_agent", ""), to_agent=s["span_data"].get("to_agent", "")
            )
            for s in spans
            if s["span_data"].get("type") == "handoff"
        ],
        intent_category=intent.get("category"),
        intent_confidence=intent.get("confidence"),
        input_tokens=sum(int(u.get("input_tokens") or 0) for u in usage),
        output_tokens=sum(int(u.get("output_tokens") or 0) for u in usage),
        generations=len(usage),
        raw_metadata=dict(metadata),
    )


def load(path: str | Path) -> list[Run]:
    return [parse(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def iter_load(path: str | Path) -> Iterator[Run]:
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield parse(json.loads(line))
