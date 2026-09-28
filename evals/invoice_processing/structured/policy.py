from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .actions import Action, Branch


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    description: str = ""

    review_cost_usd: float = 75.0
    materiality_gate: bool = True

    # (role, inclusive amount limit), ascending; the final tier is unbounded.
    approval_tiers: tuple[tuple[str, float], ...] = (
        ("department_head", 25_000),
        ("controller", 100_000),
        ("cfo", 250_000),
        ("board", float("inf")),
    )
    segregation_amount: float = 10_000  # requester may not be sole approver above this

    price_variance_pct: float = 0.02
    price_variance_floor: float = 100.0
    short_pay_max_share: float = 0.25
    short_pay_max_amount: float = 2_500.0
    adjustment_tolerance_abs: float = 250.0
    adjustment_tolerance_pct: float = 0.01
    stale_days: int = 120

    decided: float = 0.5  # Threshold for raising a problem finding.
    clear_decided: float = 0.5  # Threshold for accepting exculpatory evidence.
    state_floor: float = 0.0  # Below this probability, read the choice as unsure.

    bank_change_verified_window_days: int = 90
    first_payment_verify_amount: float = 2_500.0
    new_mailbox_callback_amount: float = 25_000.0
    owner_approval_relaxes_scope: bool = True

    vague_ceiling: float = 5_000.0  # Ignore ambiguity at or below this amount.
    vendor_only_evidence_ceiling: float = 5_000.0

    schedule_window_days: int = 7
    take_early_discount: bool = True

    def required_role(self, amount: float) -> str:
        for role, limit in self.approval_tiers:
            if amount <= limit:
                return role
        return self.approval_tiers[-1][0]


STARTUP = PolicyProfile(
    name="startup",
    description="Lean: automate ordinary ambiguity, be paranoid about payment destination.",
)

ENTERPRISE = PolicyProfile(
    name="enterprise",
    description="Controlled: auditability first, more review capacity, tight tolerances.",
    review_cost_usd=20.0,
    materiality_gate=False,
    approval_tiers=(
        ("department_head", 10_000),
        ("controller", 50_000),
        ("cfo", 250_000),
        ("board", float("inf")),
    ),
    segregation_amount=0.0,
    price_variance_pct=0.005,
    price_variance_floor=25.0,
    short_pay_max_share=0.10,
    short_pay_max_amount=1_000.0,
    adjustment_tolerance_abs=50.0,
    adjustment_tolerance_pct=0.0025,
    stale_days=60,
    first_payment_verify_amount=0.0,
    new_mailbox_callback_amount=10_000.0,
    owner_approval_relaxes_scope=False,
    vague_ceiling=1_000.0,
    vendor_only_evidence_ceiling=0.0,
)

RETAILER = PolicyProfile(
    name="high_volume_retailer",
    description="Volume: aggressive automation for established recurring vendors.",
    review_cost_usd=40.0,
    approval_tiers=(
        ("department_head", 100_000),
        ("controller", 500_000),
        ("cfo", float("inf")),
    ),
    segregation_amount=25_000.0,
    price_variance_pct=0.03,
    price_variance_floor=250.0,
    short_pay_max_share=0.50,
    short_pay_max_amount=10_000.0,
    adjustment_tolerance_abs=500.0,
    adjustment_tolerance_pct=0.02,
    stale_days=180,
    first_payment_verify_amount=10_000.0,
    new_mailbox_callback_amount=50_000.0,
    vague_ceiling=10_000.0,
    vendor_only_evidence_ceiling=10_000.0,
)

PROFILES: dict[str, PolicyProfile] = {p.name: p for p in (STARTUP, ENTERPRISE, RETAILER)}
PROFILE_ALIASES: dict[str, str] = {
    "retailer": "high_volume_retailer",
    "hvr": "high_volume_retailer",
    "lean": "startup",
    "controlled": "enterprise",
    "volume": "high_volume_retailer",
}


def canonical_profile_name(name: str) -> str:
    return PROFILE_ALIASES.get(name, name)


class RuleTrace(BaseModel):

    model_config = ConfigDict(extra="forbid")

    rule: str
    step: str
    branch: Branch
    fired: bool
    actions: list[Action] = Field(default_factory=list)
    adds: list[Action] = Field(default_factory=list, description="Actions this rule adds when it fires.")
    inputs: dict[str, Any] = Field(default_factory=dict)
    threshold: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: str
    actions: list[Action]
    primary_action: Action
    payment_blocked: bool
    payable_now: float = Field(description="Amount released by this decision (0 when blocked).")
    disputed_amount: float = 0.0
    disputed_lines: list[str] = Field(default_factory=list, description="Line refs and reasons.")
    vendor_must_fix: list[str] = Field(default_factory=list)
    waiting_on: list[str] = Field(default_factory=list)
    approval_role_required: str | None = None
    deferred_until: dt.date | None = None
    expected_loss_usd: float | None = None
    trace: list[RuleTrace]

    def explanation(self) -> str:
        head = f"[{self.policy}] -> {', '.join(a.value for a in self.actions)}"
        if self.approval_role_required:
            head += f" ({self.approval_role_required})"
        lines = [
            head,
            f"  payable_now={self.payable_now:,.2f}  disputed={self.disputed_amount:,.2f}",
        ]
        if self.deferred_until:
            lines.append(f"  deferred_until={self.deferred_until}")
        for t in self.trace:
            if t.fired and t.actions:
                inputs = ", ".join(f"{k}={_fmt(v)}" for k, v in t.inputs.items())
                thresholds = ", ".join(f"{k}={_fmt(v)}" for k, v in t.threshold.items())
                suffix = f" | policy {thresholds}" if thresholds else ""
                lines.append(f"  - [{t.step}] {t.rule}: {inputs}{suffix}")
                if t.note:
                    lines.append(f"      {t.note}")
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, str) and len(v) > 80:
        return v[:77] + "..."
    return str(v)
