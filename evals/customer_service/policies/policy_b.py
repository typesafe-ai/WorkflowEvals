"""Generous policy: favor automated assistance and retention over handoffs."""
from __future__ import annotations

from ..gates import Always, ChoiceIn, ChoiceTop, NodeRan, NoulGate, ScoreExpect, ScoreTail
from . import facts as F
from .base import (CANCEL, CLEAR_PENDING, CLOSE, DISPUTE, FREEZE, MARK_RESOLVED, REDACT, REFUND, RETENTION, SET_INTENT, WAIVE_FEE,
                   Policy, R, Rule, Section, flag, handoff, pending, priority)

CRED = 0.5
LEGAL = 0.7
PUBLIC = 0.7
HUMAN = 0.7
ABUSE_TAIL = 0.5      # P(frustration >= 4)
FRUSTR_TAIL = 0.6     # P(frustration >= 3)
ERROR = 0.6
FALSE_CLAIM = 0.7
RESOLVED = 0.6
CONSENT = 0.5
TERMS = 0.6
INTENT_CONF = 0.4
FACT = 0.6
INSTANT_MAX = 25.0
PROPOSE_MAX = 150.0
EVIDENCE_MAX = 250.0
GOODWILL_MAX = 40.0
GOODWILL_TENURE = 12
REASON_CONF = 0.8
EVIDENCE = 0.6
HARDSHIP_TAIL = 0.6   # P(hardship >= 1)
URGENCY_E, URGENCY_C = 1.5, 0.5
OFFER = 0.5
LEAVING_TAIL = 0.7    # P(churn >= 3)

ELIGIBLE_REASONS = ["damaged_or_faulty", "never_arrived", "wrong_item", "duplicate_or_erroneous_charge", "changed_mind"]
money_context = ChoiceIn("intent", ["refund_request", "billing_dispute"]) | ChoiceIn("desired_outcome", ["money_back"])
accepts = ChoiceTop("proposal_reply", CONSENT, "accepts") & ~NoulGate("changes_terms", TERMS)
declines = ChoiceTop("proposal_reply", CONSENT, "declines")
changed = ChoiceTop("proposal_reply", CONSENT, "accepts") & NoulGate("changes_terms", TERMS)
explained = NoulGate("recognizes_charge", FACT) | NoulGate("authorized_person_made_it", FACT)
takeover = NoulGate("account_or_device_anomalies", FACT)
lost = ChoiceTop("card_possession", FACT, "lost_or_stolen")

SECTIONS = [
    Section("Safety and conduct (all that apply)", [
        Rule("credentials exposed", NoulGate("shares_credentials", CRED), [REDACT, R("warn_not_to_share_credentials")], stop=False),
        Rule("legal or regulatory threat", NoulGate("threat_legal_regulatory", LEGAL), [flag("legal_threat"), handoff("compliance"), R("handoff_notice")]),
        Rule("chargeback or public threat", NoulGate("threat_chargeback_or_public", PUBLIC), [flag("public_complaint_threat")], stop=False),
        Rule("hostile", ScoreTail("frustration", 4, ABUSE_TAIL), [flag("abusive_customer"), handoff("supervisor"), R("handoff_notice")]),
        Rule("asks for a person", NoulGate("requests_human", HUMAN), [handoff("general"), R("handoff_notice")]),
        Rule("frustrated", ScoreTail("frustration", 3, FRUSTR_TAIL), [priority("high"), R("empathize_and_continue")], stop=False,
             why="Frustration is met with empathy and a higher priority, not a transfer."),
        Rule("alleges an error", NoulGate("claims_agent_error", ERROR), [flag("agent_error"), R("apologize_and_correct")], stop=False),
        Rule("assistant claimed a refund the record does not show", NoulGate("claims_refund_done", FALSE_CLAIM) & F.refund_not_issued, [flag("agent_error"), R("apologize_and_correct")], stop=False,
             why="Corrected in the conversation; the money rules below may still pay it now."),
        Rule("assistant claimed the card was secured while it is active", NoulGate("claims_secured", FALSE_CLAIM) & F.card_active, [flag("agent_error"), FREEZE, R("apologize_and_correct")], stop=False,
             why="Make the claim true: the card is frozen now."),
        Rule("urgent", ScoreExpect("urgency", URGENCY_E, URGENCY_C), [priority("high")], stop=False),
    ], mode="all"),
    Section("Answering a pending proposal", [
        Rule("refund accepted on different terms", F.pending_is("issue_refund") & changed & F.refund_eligible, [CLEAR_PENDING, handoff("refunds"), R("explain_refund_under_review")],
             why="A yes to a different amount goes to a person who can pay it."),
        Rule("accepted on different terms", changed, [R("ask_desired_outcome")]),
        Rule("refund accepted, within limit", F.pending_is("issue_refund") & accepts & F.refund_eligible & F.refund_at_most(PROPOSE_MAX), [REFUND, CLEAR_PENDING, MARK_RESOLVED, R("confirm_refund_issued")]),
        Rule("refund accepted, over limit", F.pending_is("issue_refund") & accepts & F.refund_eligible & F.refund_over(PROPOSE_MAX), [CLEAR_PENDING, handoff("refunds"), R("explain_refund_under_review")]),
        Rule("refund declined", F.pending_is("issue_refund") & declines, [CLEAR_PENDING, R("ask_desired_outcome")]),
        Rule("cancellation accepted", F.pending_is("cancel_subscription") & accepts & F.subscription_active, [CANCEL, CLEAR_PENDING, MARK_RESOLVED, R("confirm_cancellation")]),
        Rule("cancellation declined", F.pending_is("cancel_subscription") & declines, [CLEAR_PENDING, RETENTION, R("offer_retention_incentive")],
             why="A customer who hesitates to cancel is offered something to stay."),
        Rule("freeze accepted", F.pending_is("freeze_card") & accepts & F.card_active, [FREEZE, CLEAR_PENDING, R("confirm_card_frozen")]),
        Rule("freeze declined", F.pending_is("freeze_card") & declines, [CLEAR_PENDING, R("acknowledge_no_freeze")]),
        Rule("dispute accepted", F.pending_is("open_dispute") & accepts & F.has_transaction, [DISPUTE, CLEAR_PENDING, R("confirm_dispute_opened")]),
        Rule("dispute declined", F.pending_is("open_dispute") & declines, [CLEAR_PENDING, R("ask_desired_outcome")]),
        Rule("hardship referral accepted", F.pending_is("hardship_referral") & accepts, [CLEAR_PENDING, handoff("hardship"), R("handoff_notice")]),
        Rule("hardship referral declined", F.pending_is("hardship_referral") & declines, [CLEAR_PENDING, R("explain_charge_or_policy")]),
    ], intro="Read from the consent node (the proposal and the reply only)."),
    Section("Closing", [
        Rule("resolved", NoulGate("issue_resolved", RESOLVED), [MARK_RESOLVED, CLOSE, R("closing_acknowledgement")]),
    ]),
    Section("Recording the intent", [
        Rule("intent clear", ChoiceTop("intent", INTENT_CONF), [SET_INTENT], stop=False),
    ]),
    Section("Unauthorized activity (acts first, asks later)", [
        Rule("explained after all", explained, [R("explain_charge_or_policy")]),
        Rule("card lost or stolen", lost & F.card_active, [FREEZE, priority("urgent"), handoff("security"), R("confirm_card_frozen")],
             why="A freeze is reversible and cheap; it is done at once, without a proposal step."),
        Rule("account takeover", takeover & F.card_active, [FREEZE, priority("urgent"), handoff("security"), R("account_recovery_steps")]),
        Rule("charge on file", NodeRan("fraud") & NoulGate("reports_unauthorized", FACT) & F.has_transaction & F.card_active, [DISPUTE, handoff("security"), R("confirm_dispute_opened")]),
        Rule("nothing on file", NodeRan("fraud") & NoulGate("reports_unauthorized", FACT) & F.no_transaction, [R("ask_charge_details")]),
        Rule("card already frozen", NodeRan("fraud") & NoulGate("reports_unauthorized", FACT) & F.card_frozen, [handoff("security"), R("handoff_notice")]),
    ], intro="Read from the fraud node."),
    Section("Money back", [
        Rule("already compensated", money_context & (F.refund_already_issued | NoulGate("already_compensated", 0.6)), [R("explain_charge_or_policy")]),
        Rule("instant refund", money_context & F.refund_eligible & F.refund_at_most(INSTANT_MAX) & ChoiceIn("desired_outcome", ["money_back"]), [REFUND, MARK_RESOLVED, R("confirm_refund_issued")],
             why="Under $25 the cost of asking exceeds the cost of paying."),
        Rule("propose a refund", money_context & F.refund_eligible & F.refund_at_most(PROPOSE_MAX) & ChoiceIn("refund_reason", ELIGIBLE_REASONS), [pending("issue_refund"), R("propose_refund")]),
        Rule("propose a larger refund on evidence", money_context & F.refund_eligible & F.refund_at_most(EVIDENCE_MAX) & ChoiceIn("refund_reason", ELIGIBLE_REASONS) & NoulGate("offers_evidence", EVIDENCE),
             [pending("issue_refund"), R("propose_refund")], why="Photos, receipts or tracking raise the ceiling to $250."),
        Rule("propose a refund, reason unknown", money_context & F.refund_eligible & F.refund_at_most(PROPOSE_MAX), [R("ask_refund_reason")]),
        Rule("large refund", money_context & F.refund_eligible, [handoff("refunds"), R("explain_refund_under_review")]),
        Rule("goodwill refund", money_context & F.refund_ineligible & F.refund_at_most(GOODWILL_MAX) & F.tenure_at_least(GOODWILL_TENURE) & ChoiceTop("refund_reason", REASON_CONF),
             [REFUND, MARK_RESOLVED, R("confirm_refund_issued")], why="A tenured customer with a clear reason is paid even when the claim is outside policy."),
        Rule("not eligible", money_context & F.refund_ineligible, [R("explain_refund_not_eligible")]),
        Rule("waive the fee", money_context & F.has_fee & ScoreTail("hardship", 1, HARDSHIP_TAIL), [WAIVE_FEE, MARK_RESOLVED, R("confirm_refund_issued")],
             why="Any mention of money being tight waives a fee."),
        Rule("nothing on file", money_context & F.no_refund_claim, [R("ask_charge_details")]),
    ], intro="Read from the money node."),
    Section("Keeping the customer", [
        Rule("leaving, not open to an offer", ChoiceIn("intent", ["cancel_account"]) & F.subscription_active & ScoreTail("churn_risk", 3, LEAVING_TAIL) & ~NoulGate("open_to_offer", OFFER),
             [pending("cancel_subscription"), R("propose_cancellation")]),
        Rule("open to an offer", NoulGate("open_to_offer", OFFER), [RETENTION, R("offer_retention_incentive")]),
        Rule("at risk", ScoreTail("churn_risk", 2, 0.5), [RETENTION, R("offer_retention_incentive")]),
        Rule("leaving over price", ChoiceIn("intent", ["cancel_account"]) & ChoiceIn("cancellation_reason", ["price"]), [RETENTION, R("offer_retention_incentive")]),
    ], intro="Read from the retention node, which adds the subscription record."),
    Section("Other issues", [
        Rule("locked out", ChoiceIn("intent", ["account_access"]), [R("account_recovery_steps")]),
        Rule("card declined", ChoiceIn("intent", ["card_declined"]), [R("explain_charge_or_policy")]),
        Rule("where is my order", ChoiceIn("intent", ["delivery_issue"]) & F.has_order, [R("provide_order_status")]),
        Rule("which order", ChoiceIn("intent", ["delivery_issue"]) & F.no_order, [R("ask_order_details")]),
        Rule("wants to cancel", ChoiceIn("intent", ["cancel_account"]), [R("ask_cancellation_reason")]),
        Rule("intent unclear", ~ChoiceTop("intent", INTENT_CONF), [R("ask_clarify_issue")]),
        Rule("general", Always(), [R("explain_charge_or_policy")]),
    ]),
]

POLICY_B = Policy(
    key="B", name="Policy B — Generous (online retailer, recall first)",
    summary=(
        "Policy B is written for an online retailer whose worst outcome is a customer who leaves. The assistant keeps the "
        "conversation whenever it can: frustration earns empathy and a higher priority rather than a transfer, and a person "
        "is brought in only for hostility, an explicit request at high confidence, or a legal threat. Money moves easily: "
        "refunds up to $25 are paid on the spot, up to $150 on a yes, up to $250 when evidence is offered, and tenured "
        "customers receive goodwill refunds up to $40 even outside policy; any mention of money being tight waives a fee. "
        "A yes on different terms is sent to a person who can pay it. Fraud is handled by acting first (freeze, dispute) and "
        "explaining after; investigation facts count at 0.6. A false claim by the assistant is corrected in place, and a "
        "false 'your card is secured' is made true by freezing it. Any opening for retention earns an offer. The safety "
        "section fires every matching rule, so several actions per turn are normal."
    ),
    sections=SECTIONS,
)
