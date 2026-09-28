from __future__ import annotations

from dataclasses import dataclass

from .trace import Run, ToolCall

# Transient runtime errors indicate infrastructure failures.
TRANSIENT_ERROR_CODES = frozenset(
    {
        "GATEWAY_TIMEOUT",
        "UPSTREAM_TIMEOUT",
        "TIMEOUT",
        "RATE_LIMITED",
        "SERVICE_UNAVAILABLE",
        "UPSTREAM_UNAVAILABLE",
        "CONNECTION_RESET",
        "AUTH_EXPIRED",
    }
)

# These error codes corroborate tool misuse; attribution still comes from the model.
CALLER_ERROR_CODES = frozenset(
    {"TOOL_NOT_FOUND", "MALFORMED_TOOL_CALL", "INVALID_ARGUMENTS", "SCHEMA_MISMATCH"}
)

RESEARCH = "research"
ACTION = "action"


@dataclass(frozen=True)
class Facts:

    consequential_calls: tuple[ToolCall, ...]
    destructive_calls: tuple[ToolCall, ...]
    unentitled_calls: tuple[ToolCall, ...]

    failed_calls: tuple[ToolCall, ...]
    error_codes: tuple[str, ...]
    transient_only: bool
    caller_errors: bool
    retried_after_failure: bool

    intent_category: str | None
    intent_confidence: float | None

    has_reaction: bool
    reaction_signal: str | None
    csat: int | None
    reopened: bool

    # Routing metadata is reported but never used by gates.
    duration_s: float | None
    input_tokens: int
    output_tokens: int
    tool_calls: int
    generations: int
    guardrail_triggered: bool
    handed_off: bool
    authenticated: bool | None
    customer_segment: str | None

    @property
    def touched_consequential(self) -> bool:
        return bool(self.consequential_calls)

    @property
    def is_research(self) -> bool:
        return self.intent_category == RESEARCH

    @property
    def had_failure(self) -> bool:
        return bool(self.failed_calls)

    def to_json(self) -> dict:
        return {
            "consequential_calls": [c.label for c in self.consequential_calls],
            "destructive_calls": [c.label for c in self.destructive_calls],
            "unentitled_calls": [c.label for c in self.unentitled_calls],
            "failed_calls": [c.label for c in self.failed_calls],
            "error_codes": list(self.error_codes),
            "transient_only": self.transient_only,
            "caller_errors": self.caller_errors,
            "retried_after_failure": self.retried_after_failure,
            "intent_category": self.intent_category,
            "intent_confidence": self.intent_confidence,
            "has_reaction": self.has_reaction,
            "reaction_signal": self.reaction_signal,
            "csat": self.csat,
            "reopened": self.reopened,
            "duration_s": self.duration_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "generations": self.generations,
            "guardrail_triggered": self.guardrail_triggered,
            "handed_off": self.handed_off,
            "authenticated": self.authenticated,
            "customer_segment": self.customer_segment,
        }


def _retried_after_failure(run: Run) -> bool:
    first_failure: dict[str, int] = {}
    for call in run.calls:
        if call.failed and call.name not in first_failure:
            first_failure[call.name] = call.ordinal
    return any(call.ordinal > first_failure.get(call.name, call.ordinal) for call in run.calls)


def read(run: Run) -> Facts:
    failed = tuple(run.failed_calls)
    codes = tuple(c.error_code for c in failed if c.error_code)
    reaction = run.reaction
    authenticated = run.context.get("authenticated")
    return Facts(
        consequential_calls=tuple(run.consequential_calls),
        destructive_calls=tuple(run.destructive_calls),
        unentitled_calls=tuple(c for c in run.calls if (g := run.grant_for(c)) and not g.entitled),
        failed_calls=failed,
        error_codes=codes,
        transient_only=bool(codes) and all(code in TRANSIENT_ERROR_CODES for code in codes),
        caller_errors=any(code in CALLER_ERROR_CODES for code in codes),
        retried_after_failure=_retried_after_failure(run),
        intent_category=run.intent_category,
        intent_confidence=run.intent_confidence,
        has_reaction=bool(reaction) and not reaction.is_silent,
        reaction_signal=reaction.signal if reaction else None,
        csat=reaction.csat if reaction else None,
        reopened=bool(reaction and reaction.reopened),
        duration_s=run.duration_s,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        tool_calls=len(run.calls),
        generations=run.generations,
        guardrail_triggered=any(g.triggered for g in run.guardrails),
        handed_off=bool(run.handoffs),
        authenticated=bool(authenticated) if isinstance(authenticated, bool) else None,
        customer_segment=run.context.get("customer_segment"),
    )
