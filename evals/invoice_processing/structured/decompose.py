from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .actions import apply_precedence, parse_action
from .facts import ExactFacts, compute_exact_facts, resolve_facts
from .findings import Findings
from .findings import from_signals as findings_from_signals
from .flow import decide
from .policy import PROFILES, Decision, PolicyProfile, canonical_profile_name
from .questions import plan_questions
from .semantic import Signals, SystemOneLike, ask_signals
from .state import Packet, Sidecar


class QuestionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asked: dict[str, str] = Field(description="question id -> instruction text")
    skipped: dict[str, str] = Field(description="question id -> reason")
    axes: dict[str, str] = Field(description="question id -> axis")


class ProfileOutcome(BaseModel):

    model_config = ConfigDict(extra="forbid")

    expected: list[str]
    actual: list[str]
    exact_match: bool = Field(description="Expected and actual action sets are identical.")
    primary_match: bool = Field(
        description="The highest-precedence expected action equals the actual primary action."
    )
    missing: list[str] = Field(default_factory=list)
    unexpected: list[str] = Field(default_factory=list)


class LabelComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by_profile: dict[str, ProfileOutcome] = Field(default_factory=dict)

    @property
    def all_exact(self) -> bool:
        return all(o.exact_match for o in self.by_profile.values())


class FactAgreement(BaseModel):
    """Compare derived facts with reference values on overlapping keys only."""

    model_config = ConfigDict(extra="forbid")

    compared: int
    disagreements: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="key -> {ours, reference}"
    )


class Decomposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    family: str | None
    company_id: str
    created_at: datetime
    exact_facts: ExactFacts
    question_plan: QuestionPlan
    signals: Signals | None = Field(default=None, description="None for an offline (facts-only) run.")
    findings: Findings | None = Field(default=None, description="The answers read as findings (Part A).")
    decisions: dict[str, Decision] = Field(default_factory=dict)
    label_comparison: LabelComparison | None = None
    fact_agreement: FactAgreement | None = None


def _plan(packet: Packet, facts: ExactFacts) -> QuestionPlan:
    asked, skipped, axes = plan_questions(packet, facts)
    return QuestionPlan(
        asked={qid: str(q.instructions) for qid, q in asked.items()},
        skipped=skipped,
        axes=axes,
    )


def _compare_to_sidecar(sidecar: Sidecar | None, decisions: dict[str, Decision]) -> LabelComparison | None:
    if sidecar is None or not sidecar.actions_by_profile:
        return None
    by_canonical = {canonical_profile_name(k): v for k, v in sidecar.actions_by_profile.items()}
    outcomes: dict[str, ProfileOutcome] = {}
    for profile, decision in decisions.items():
        expected_spec = by_canonical.get(profile) or by_canonical.get("*")
        if expected_spec is None:
            continue
        expected = apply_precedence({parse_action(name) for name in expected_spec.actions})
        expected_primary = parse_action(expected_spec.primary) if expected_spec.primary else expected[0]
        actual = decision.actions
        outcomes[profile] = ProfileOutcome(
            expected=[a.value for a in expected],
            actual=[a.value for a in actual],
            exact_match=set(expected) == set(actual),
            primary_match=expected_primary == decision.primary_action,
            missing=sorted(a.value for a in set(expected) - set(actual)),
            unexpected=sorted(a.value for a in set(actual) - set(expected)),
        )
    return LabelComparison(by_profile=outcomes)


# Reference field -> (local field, transform applied to the local value).
FACT_ALIASES: dict[str, tuple[str, Any]] = {
    "as_of_date": ("as_of", None),
    "invoice_amount": ("invoice_total", None),
    "line_items_sum_to_total": ("lines_sum_to_total", None),
    "po_cited_on_invoice": ("po_referenced", None),
    "po_number_matches": ("po_referenced_matches", None),
    "po_overrun_amount": ("overrun_amount", None),
    "bill_to_matches_po_entity": ("bill_to_entity_matches_po", None),
    "bill_to_matches_po_department": ("bill_to_department_matches_po", None),
    "bill_to_is_company_entity": ("bill_to_entity_is_company_entity", None),
    "bank_holder_matches_vendor": (
        "account_holder_differs",
        lambda v: None if v is None else not v,
    ),
    "bank_verified_on_file": ("bank_change_verified_on_file", None),
    "prior_invoice_with_same_amount": ("similar_amount_seen_before", None),
    "prior_payment_count": ("paid_invoice_count", None),
    "vendor_disputes": ("disputed_invoice_count", None),
    "vendor_reversals": ("reversed_invoice_count", None),
    "any_external_sender_off_file": ("sender_domains_off_file", lambda v: bool(v)),
    "required_approval_present": ("purchase_approved", None),
    "invoice_approval_status": ("invoice_approval_status", None),
    "past_due_on_receipt": ("is_overdue", None),
}


def _compare_facts(sidecar: Sidecar | None, facts: ExactFacts) -> FactAgreement | None:
    if sidecar is None or not sidecar.exact_facts:
        return None
    ours = facts.model_dump(mode="json")
    disagreements = {}
    compared = 0
    for key, reference in sidecar.exact_facts.items():
        if key in FACT_ALIASES:
            our_key, transform = FACT_ALIASES[key]
            mine = ours[our_key]
            if transform is not None:
                mine = transform(mine)
        elif key in ours:
            mine = ours[key]
        else:
            continue
        compared += 1
        numeric = (
            isinstance(mine, (int, float))
            and isinstance(reference, (int, float))
            and not isinstance(mine, bool)
            and not isinstance(reference, bool)
        )
        same = abs(mine - reference) < 0.01 if numeric else mine == reference
        if not same:
            disagreements[key] = {"ours": mine, "reference": reference}
    return FactAgreement(compared=compared, disagreements=disagreements)


def _finish(
    packet: Packet,
    sidecar: Sidecar | None,
    facts: ExactFacts,
    signals: Signals,
    profiles: Iterable[PolicyProfile],
) -> Decomposition:
    profiles = list(profiles)
    findings = findings_from_signals(
        facts, signals
    )
    plan = _plan(packet, facts)
    decisions = {}
    for prof in profiles:
        # Resolve judgment-dependent facts separately for each profile.
        f_p = findings_from_signals(facts, signals, prof)
        decisions[prof.name] = decide(resolve_facts(facts, f_p), f_p, prof)
    facts = resolve_facts(facts, findings)
    return Decomposition(
        case_id=packet.case_id,
        family=sidecar.family if sidecar else None,
        company_id=packet.company.company_id,
        created_at=datetime.now(UTC),
        exact_facts=facts,
        question_plan=plan,
        signals=signals,
        findings=findings,
        decisions=decisions,
        label_comparison=_compare_to_sidecar(sidecar, decisions),
        fact_agreement=_compare_facts(sidecar, facts),
    )


def run_case(
    packet: Packet,
    client: SystemOneLike,
    model: str,
    profiles: Iterable[PolicyProfile] | None = None,
    sidecar: Sidecar | None = None,
    as_of: dt.date | None = None,
    thinking: str | None = None,
) -> Decomposition:
    facts = compute_exact_facts(packet, as_of)
    signals = ask_signals(client, model, packet, facts)
    if thinking is not None:
        signals.usage["thinking"] = thinking
    return _finish(packet, sidecar, facts, signals, profiles or PROFILES.values())
