"""Actions are {"action": name, **params}; response messages are named intents, not free text."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

RESPONSES = [
    "ask_clarify_issue",
    "ask_card_in_possession",
    "ask_charge_details",
    "ask_order_details",
    "propose_card_freeze",
    "confirm_card_frozen",
    "acknowledge_no_freeze",
    "ask_refund_reason",
    "propose_refund",
    "confirm_refund_issued",
    "explain_refund_not_eligible",
    "explain_refund_under_review",
    "ask_desired_outcome",
    "explain_charge_or_policy",
    "provide_order_status",
    "offer_retention_incentive",
    "ask_cancellation_reason",
    "propose_cancellation",
    "confirm_cancellation",
    "handoff_notice",
    "empathize_and_continue",
    "warn_not_to_share_credentials",
    "account_recovery_steps",
    "offer_hardship_options",
    "explain_dispute_process",
    "confirm_dispute_opened",
    "apologize_and_correct",
    "closing_acknowledgement",
]

QUEUES = ["security", "compliance", "refunds", "retention", "hardship", "supervisor", "general"]
REVIEW_REASONS = ["agent_error", "legal_threat", "public_complaint_threat", "abusive_customer", "ambiguous_intent", "credentials_exposed", "possible_fraud_unconfirmed", "high_value_refund"]
PRIORITIES = ["normal", "high", "urgent"]
PENDING = ["freeze_card", "issue_refund", "cancel_subscription", "open_dispute", "hardship_referral"]

SIDE_EFFECTS = ["freeze_card", "issue_refund", "open_dispute", "handoff_human", "flag_for_review", "redact_transcript",
                "apply_retention_offer", "cancel_subscription", "close_ticket"]
STATE_UPDATES = ["set_intent", "set_pending", "clear_pending", "set_priority", "mark_resolved"]
ALL_ACTIONS = ["respond_with"] + SIDE_EFFECTS + STATE_UPDATES

Action = Dict[str, Any]


def respond_with(message: str) -> Action:
    assert message in RESPONSES, message
    return {"action": "respond_with", "message": message}


def freeze_card(card_id: str) -> Action:
    return {"action": "freeze_card", "card_id": card_id}


def issue_refund(amount_usd: float) -> Action:
    return {"action": "issue_refund", "amount_usd": round(float(amount_usd), 2)}


def open_dispute(transaction_id: str) -> Action:
    return {"action": "open_dispute", "transaction_id": transaction_id}


def handoff_human(queue: str) -> Action:
    assert queue in QUEUES, queue
    return {"action": "handoff_human", "queue": queue}


def flag_for_review(reason: str) -> Action:
    assert reason in REVIEW_REASONS, reason
    return {"action": "flag_for_review", "reason": reason}


def redact_transcript() -> Action:
    return {"action": "redact_transcript"}


def apply_retention_offer() -> Action:
    return {"action": "apply_retention_offer"}


def cancel_subscription() -> Action:
    return {"action": "cancel_subscription"}


def close_ticket() -> Action:
    return {"action": "close_ticket"}


def set_intent(intent: str) -> Action:
    return {"action": "set_intent", "intent": intent}


def set_pending(proposal: str) -> Action:
    assert proposal in PENDING, proposal
    return {"action": "set_pending", "proposal": proposal}


def clear_pending() -> Action:
    return {"action": "clear_pending"}


def set_priority(level: str) -> Action:
    assert level in PRIORITIES, level
    return {"action": "set_priority", "level": level}


def mark_resolved() -> Action:
    return {"action": "mark_resolved"}


def canonical(actions: Iterable[Action]) -> List[str]:
    """Order-free, duplicate-free action comparison key."""
    return sorted({json.dumps(a, sort_keys=True) for a in actions})


def same(a: Iterable[Action], b: Iterable[Action]) -> bool:
    return canonical(a) == canonical(b)
