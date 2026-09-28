from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    PAY = "PAY"
    SCHEDULE_PAYMENT = "SCHEDULE_PAYMENT"
    SHORT_PAY = "SHORT_PAY"
    REQUEST_CORRECTED_INVOICE = "REQUEST_CORRECTED_INVOICE"
    DISPUTE_LINES = "DISPUTE_LINES"
    HOLD_REQUEST_DOCUMENTS = "HOLD_REQUEST_DOCUMENTS"
    PROJECT_OWNER_REVIEW = "PROJECT_OWNER_REVIEW"
    PROCUREMENT_REVIEW = "PROCUREMENT_REVIEW"
    APPROVAL_ESCALATION = "APPROVAL_ESCALATION"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"
    DUPLICATE_REVIEW = "DUPLICATE_REVIEW"
    REJECT_AS_DUPLICATE = "REJECT_AS_DUPLICATE"
    VERIFY_VENDOR_OUT_OF_BAND = "VERIFY_VENDOR_OUT_OF_BAND"
    HOLD_VERIFY_BANK_DETAILS = "HOLD_VERIFY_BANK_DETAILS"
    FRAUD_REVIEW = "FRAUD_REVIEW"
    REJECT_AS_UNAUTHORIZED = "REJECT_AS_UNAUTHORIZED"


class Branch(StrEnum):

    FRAUD = "fraud"
    DUPLICATE = "duplicate"
    ENTITLEMENT = "entitlement"
    DOCUMENTS = "documents"
    PROCUREMENT = "procurement"
    APPROVAL = "approval"
    SCHEDULE = "schedule"
    PAY = "pay"


BRANCH_OF: dict[Action, Branch] = {
    Action.FRAUD_REVIEW: Branch.FRAUD,
    Action.VERIFY_VENDOR_OUT_OF_BAND: Branch.FRAUD,
    Action.HOLD_VERIFY_BANK_DETAILS: Branch.FRAUD,
    Action.REJECT_AS_DUPLICATE: Branch.DUPLICATE,
    Action.DUPLICATE_REVIEW: Branch.DUPLICATE,
    Action.REJECT_AS_UNAUTHORIZED: Branch.ENTITLEMENT,
    Action.PROJECT_OWNER_REVIEW: Branch.ENTITLEMENT,
    Action.DISPUTE_LINES: Branch.ENTITLEMENT,
    Action.HOLD_REQUEST_DOCUMENTS: Branch.DOCUMENTS,
    Action.REQUEST_CORRECTED_INVOICE: Branch.DOCUMENTS,
    Action.PROCUREMENT_REVIEW: Branch.PROCUREMENT,
    Action.APPROVAL_ESCALATION: Branch.APPROVAL,
    Action.HUMAN_APPROVAL: Branch.APPROVAL,
    Action.SHORT_PAY: Branch.SCHEDULE,
    Action.SCHEDULE_PAYMENT: Branch.SCHEDULE,
    Action.PAY: Branch.PAY,
}

_PRECEDENCE: list[Action] = [
    Action.FRAUD_REVIEW,
    Action.REJECT_AS_DUPLICATE,
    Action.HOLD_VERIFY_BANK_DETAILS,
    Action.VERIFY_VENDOR_OUT_OF_BAND,
    Action.DUPLICATE_REVIEW,
    Action.REJECT_AS_UNAUTHORIZED,
    Action.PROJECT_OWNER_REVIEW,
    Action.DISPUTE_LINES,
    Action.HOLD_REQUEST_DOCUMENTS,
    Action.REQUEST_CORRECTED_INVOICE,
    Action.PROCUREMENT_REVIEW,
    Action.APPROVAL_ESCALATION,
    Action.HUMAN_APPROVAL,
    Action.SHORT_PAY,
    Action.SCHEDULE_PAYMENT,
    Action.PAY,
]
PRECEDENCE_RANK: dict[Action, int] = {action: rank for rank, action in enumerate(_PRECEDENCE)}

ACTION_ALIASES: dict[str, str] = {
    "PARTIAL_PAY": "SHORT_PAY",
    "HOLD_REQUEST_DOCUMENTATION": "HOLD_REQUEST_DOCUMENTS",
    "HOLD_VERIFY_BANK": "HOLD_VERIFY_BANK_DETAILS",
    "HOLD_BANK": "HOLD_VERIFY_BANK_DETAILS",
    "OUT_OF_BAND_VENDOR_VERIFICATION": "VERIFY_VENDOR_OUT_OF_BAND",
    "VERIFY_VENDOR_BY_PHONE": "VERIFY_VENDOR_OUT_OF_BAND",
    "SCHEDULE": "SCHEDULE_PAYMENT",
    "DUPLICATE": "REJECT_AS_DUPLICATE",
    "REJECT": "REJECT_AS_UNAUTHORIZED",
    "OWNER_REVIEW": "PROJECT_OWNER_REVIEW",
    "ESCALATE_FOR_APPROVAL": "APPROVAL_ESCALATION",
    "ROUTE_FOR_APPROVAL": "HUMAN_APPROVAL",
    "HOLD_FOR_DOCUMENTS": "HOLD_REQUEST_DOCUMENTS",
}


def parse_action(name: str) -> Action:
    key = name.strip().upper().replace(" ", "_").replace("-", "_")
    key = key.split("(")[0].rstrip("_")
    return Action(ACTION_ALIASES.get(key, key))


RELEASES_PAYMENT: frozenset[Action] = frozenset({Action.PAY, Action.SCHEDULE_PAYMENT, Action.SHORT_PAY})
VENDOR_FACING: frozenset[Action] = frozenset({Action.REQUEST_CORRECTED_INVOICE, Action.DISPUTE_LINES})
HARD_BLOCKERS: frozenset[Action] = frozenset(
    a for a in Action if a not in RELEASES_PAYMENT and a not in VENDOR_FACING
)


def apply_precedence(actions: set[Action]) -> list[Action]:
    """Hard blockers forbid all payment; vendor corrections still permit SHORT_PAY."""
    chosen = set(actions)
    if not chosen:
        return [Action.PAY]
    if chosen & HARD_BLOCKERS:
        chosen -= RELEASES_PAYMENT
    elif chosen & VENDOR_FACING:
        chosen -= {Action.PAY, Action.SCHEDULE_PAYMENT}
    if Action.SHORT_PAY in chosen:
        chosen -= {Action.PAY, Action.SCHEDULE_PAYMENT}
    if Action.PAY in chosen and Action.SCHEDULE_PAYMENT in chosen:
        chosen.discard(Action.PAY)
    return sorted(chosen, key=PRECEDENCE_RANK.__getitem__)


def payment_blocked(actions: list[Action]) -> bool:
    return not any(a in RELEASES_PAYMENT for a in actions)


DISPLAY: dict[Action, str] = {
    Action.PAY: "PAY",
    Action.SCHEDULE_PAYMENT: "SCHEDULE",
    Action.SHORT_PAY: "SHORT PAY",
    Action.REQUEST_CORRECTED_INVOICE: "REQUEST CORRECTED INVOICE",
    Action.DISPUTE_LINES: "DISPUTE LINES",
    Action.HOLD_REQUEST_DOCUMENTS: "HOLD FOR DOCUMENTS",
    Action.PROJECT_OWNER_REVIEW: "OWNER REVIEW",
    Action.PROCUREMENT_REVIEW: "PROCUREMENT REVIEW",
    Action.APPROVAL_ESCALATION: "ESCALATE FOR APPROVAL",
    Action.HUMAN_APPROVAL: "ROUTE FOR APPROVAL",
    Action.DUPLICATE_REVIEW: "DUPLICATE REVIEW",
    Action.REJECT_AS_DUPLICATE: "DUPLICATE",
    Action.VERIFY_VENDOR_OUT_OF_BAND: "VERIFY VENDOR BY PHONE",
    Action.HOLD_VERIFY_BANK_DETAILS: "HOLD BANK",
    Action.FRAUD_REVIEW: "FRAUD REVIEW",
    Action.REJECT_AS_UNAUTHORIZED: "REJECT",
}
