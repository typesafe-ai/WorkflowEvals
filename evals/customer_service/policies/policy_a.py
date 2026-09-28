"""Guarded policy: require explicit consent and hand off uncertain or costly actions."""
from __future__ import annotations

from ..gates import Always, ChoiceIn, ChoiceTop, NodeRan, NoulGate, ScoreExpect, ScoreTail
from . import facts as F
from .base import (CANCEL, CLEAR_PENDING, CLOSE, DISPUTE, FREEZE, MARK_RESOLVED, REDACT, REFUND, SET_INTENT, Policy, R, Rule, Section,
                   flag, handoff, pending, priority)

CRED = 0.5
LEGAL = 0.5
PUBLIC = 0.6
HUMAN = 0.5
ABUSE_TAIL = 0.5      # P(frustration >= 4)
FRUSTR_TAIL = 0.35    # P(frustration >= 3)
ERROR = 0.5
FALSE_CLAIM = 0.6
RESOLVED = 0.7
CONSENT = 0.6
TERMS = 0.5
INTENT_CONF = 0.6
FACT = 0.7
REFUND_MAX = 50.0
REASON_CONF = 0.5
HARDSHIP_TAIL = 0.5   # P(hardship >= 2)
URGENT_TAIL = 0.4     # P(urgency >= 3)
CHURN_E, CHURN_C = 2.5, 0.6

ELIGIBLE_REASONS = ["damaged_or_faulty", "never_arrived", "wrong_item", "duplicate_or_erroneous_charge"]
money_context = ChoiceIn("intent", ["refund_request", "billing_dispute"]) | ChoiceIn("desired_outcome", ["money_back"])
accepts = ChoiceTop("proposal_reply", CONSENT, "accepts") & ~NoulGate("changes_terms", TERMS)
declines = ChoiceTop("proposal_reply", CONSENT, "declines")
changed = ChoiceTop("proposal_reply", CONSENT, "accepts") & NoulGate("changes_terms", TERMS)

explained = NoulGate("recognizes_charge", FACT) | NoulGate("authorized_person_made_it", FACT)
takeover = NoulGate("account_or_device_anomalies", FACT)
lost = ChoiceTop("card_possession", FACT, "lost_or_stolen")
in_hand = ChoiceTop("card_possession", FACT, "has_it")
possession_unknown = NodeRan("fraud") & ~lost & ~in_hand   # Negation requires evidence that the investigation ran.

SECTIONS = [
    Section("Safety, checked before anything else", [
        Rule("credentials exposed", NoulGate("shares_credentials", CRED), [REDACT, R("warn_not_to_share_credentials")],
             why="A secret in the transcript is removed first; the issue is handled on the next turn."),
        Rule("legal or regulatory threat", NoulGate("threat_legal_regulatory", LEGAL), [handoff("compliance"), flag("legal_threat"), R("handoff_notice")],
             why="Compliance owns anything that may become a regulator complaint."),
        Rule("chargeback or public threat", NoulGate("threat_chargeback_or_public", PUBLIC), [flag("public_complaint_threat")], stop=False,
             why="Noted for review; the conversation continues."),
    ]),
    Section("What the assistant claimed, against the record", [
        Rule("claimed a refund the record does not show", NoulGate("claims_refund_done", FALSE_CLAIM) & F.refund_not_issued,
             [flag("agent_error"), handoff("supervisor"), R("apologize_and_correct")],
             why="A false 'your refund has been issued' is corrected by a person, not repeated by the bot."),
        Rule("claimed the card was secured while it is active", NoulGate("claims_secured", FALSE_CLAIM) & F.card_active,
             [flag("agent_error"), handoff("security"), R("apologize_and_correct")]),
    ], intro="These read the integrity node, which runs only when a fraud or money node ran and the assistant spoke since the previous customer message."),
    Section("Answering a pending proposal", [
        Rule("accepted, but on different terms", changed, [R("ask_desired_outcome")],
             why="A yes to a different amount or action is a new request, not consent. The proposal stays pending."),
        Rule("freeze accepted", F.pending_is("freeze_card") & accepts & F.card_active, [FREEZE, CLEAR_PENDING, R("confirm_card_frozen"), handoff("security")],
             why="The freeze executes on a yes; the dispute itself is a person's job."),
        Rule("freeze declined", F.pending_is("freeze_card") & declines, [CLEAR_PENDING, R("acknowledge_no_freeze"), handoff("security")],
             why="An unprotected card with a live fraud claim still needs a person."),
        Rule("refund accepted, small", F.pending_is("issue_refund") & accepts & F.refund_eligible & F.refund_at_most(REFUND_MAX),
             [REFUND, CLEAR_PENDING, R("confirm_refund_issued"), MARK_RESOLVED]),
        Rule("refund accepted, large", F.pending_is("issue_refund") & accepts & F.refund_eligible & F.refund_over(REFUND_MAX),
             [CLEAR_PENDING, handoff("refunds"), R("explain_refund_under_review")]),
        Rule("refund declined", F.pending_is("issue_refund") & declines, [CLEAR_PENDING, R("ask_desired_outcome")]),
        Rule("cancellation accepted", F.pending_is("cancel_subscription") & accepts & F.subscription_active, [CANCEL, CLEAR_PENDING, R("confirm_cancellation"), MARK_RESOLVED]),
        Rule("cancellation declined", F.pending_is("cancel_subscription") & declines, [CLEAR_PENDING, R("closing_acknowledgement")]),
        Rule("dispute accepted", F.pending_is("open_dispute") & accepts & F.has_transaction, [DISPUTE, CLEAR_PENDING, R("confirm_dispute_opened"), handoff("security")]),
        Rule("dispute declined", F.pending_is("open_dispute") & declines, [CLEAR_PENDING, R("ask_desired_outcome")]),
        Rule("hardship referral accepted", F.pending_is("hardship_referral") & accepts, [CLEAR_PENDING, handoff("hardship"), R("handoff_notice")]),
        Rule("hardship referral declined", F.pending_is("hardship_referral") & declines, [CLEAR_PENDING, R("explain_charge_or_policy")]),
    ], intro="Read from the consent node, which sees only the proposal and the reply. A proposal stays pending until the customer clearly accepts or declines it as stated; an unclear reply falls through with the proposal still pending."),
    Section("Handing off to a person", [
        Rule("asks for a person", NoulGate("requests_human", HUMAN), [handoff("general"), R("handoff_notice")]),
        Rule("hostile", ScoreTail("frustration", 4, ABUSE_TAIL), [handoff("supervisor"), flag("abusive_customer"), R("handoff_notice")]),
        Rule("angry", ScoreTail("frustration", 3, FRUSTR_TAIL), [handoff("supervisor"), R("handoff_notice")],
             why="A low bar: a third of the mass at 'angry' or above is enough."),
        Rule("alleges an error", NoulGate("claims_agent_error", ERROR), [flag("agent_error"), handoff("supervisor"), R("apologize_and_correct")]),
    ]),
    Section("Closing", [
        Rule("resolved", NoulGate("issue_resolved", RESOLVED) & F.no_pending, [MARK_RESOLVED, CLOSE, R("closing_acknowledgement")]),
    ]),
    Section("Recording the intent", [
        Rule("intent unclear", ~ChoiceTop("intent", INTENT_CONF), [R("ask_clarify_issue")],
             why="Nothing is acted on until the model is confident what the customer wants, after the two-way check if one ran."),
        Rule("intent clear", ChoiceTop("intent", INTENT_CONF), [SET_INTENT], stop=False),
    ]),
    Section("Unauthorized activity: the investigation tree", [
        Rule("explained after all", explained, [R("explain_charge_or_policy")],
             why="The customer recognises the charge, or someone with access made it. Nothing to protect."),
        Rule("card lost or stolen", lost & F.card_active, [pending("freeze_card"), priority("urgent"), R("propose_card_freeze")]),
        Rule("account takeover", takeover & F.card_active, [pending("freeze_card"), priority("urgent"), handoff("security"), R("propose_card_freeze")],
             why="Login codes, a changed email, an unknown device: the account, not just the card, is at risk."),
        Rule("possession not stated", possession_unknown & F.card_active, [R("ask_card_in_possession")]),
        Rule("card in hand, details do not match", in_hand & ~NoulGate("details_match_ledger", FACT), [R("ask_charge_details")]),
        Rule("card in hand, card-present charge", in_hand & F.card_active & F.channel_is("card_present", "atm"), [pending("freeze_card"), priority("high"), R("propose_card_freeze")],
             why="Card present and card in hand: a counterfeit. The card is replaced."),
        Rule("card in hand, remote charge", in_hand & F.card_active & F.channel_is("online", "recurring"), [pending("freeze_card"), priority("high"), R("propose_card_freeze")],
             why="Stolen credentials are still stolen; the number is replaced."),
        Rule("card in hand, nothing on file", in_hand & F.no_transaction, [R("ask_charge_details")]),
        Rule("card already frozen", NoulGate("reports_unauthorized", 0.5) & F.card_frozen, [handoff("security"), R("handoff_notice")]),
    ], intro="Read from the fraud node, which adds the ledger entry to the state. Rules are walked in order; the first established branch wins."),
    Section("Money back", [
        Rule("hardship", money_context & ScoreTail("hardship", 2, HARDSHIP_TAIL) & F.has_fee, [pending("hardship_referral"), R("offer_hardship_options")],
             why="A fee dispute with real hardship goes to the hardship team, with consent."),
        Rule("already compensated", money_context & (F.refund_already_issued | NoulGate("already_compensated", 0.6)), [R("explain_charge_or_policy")]),
        Rule("asks for more than the record", money_context & F.refund_eligible & ChoiceTop("amount_vs_record", 0.5, "more"), [handoff("refunds"), R("explain_refund_under_review")],
             why="Only a person can pay more than the amount on file."),
        Rule("no reason yet", money_context & F.refund_eligible & ChoiceIn("refund_reason", ["no_reason_given"]), [R("ask_refund_reason")]),
        Rule("propose a small refund", money_context & F.refund_eligible & F.refund_at_most(REFUND_MAX) & ChoiceIn("refund_reason", ELIGIBLE_REASONS) & ChoiceTop("refund_reason", REASON_CONF),
             [pending("issue_refund"), R("propose_refund")]),
        Rule("refund needs a person", money_context & F.refund_eligible, [handoff("refunds"), R("explain_refund_under_review")],
             why="Over $50, a doubtful reason, or a reason outside the eligible set: a person decides."),
        Rule("not eligible", money_context & F.refund_ineligible, [R("explain_refund_not_eligible")]),
        Rule("nothing on file", money_context & F.no_refund_claim, [R("ask_charge_details")]),
    ], intro="Read from the money node, which adds the claim or fee on file to the state."),
    Section("Other issues", [
        Rule("locked out", ChoiceIn("intent", ["account_access"]), [R("account_recovery_steps")]),
        Rule("card declined, stranded", ChoiceIn("intent", ["card_declined"]) & ScoreTail("urgency", 3, URGENT_TAIL), [priority("urgent"), handoff("general"), R("handoff_notice")]),
        Rule("card declined", ChoiceIn("intent", ["card_declined"]), [R("explain_charge_or_policy")]),
        Rule("wants to cancel", ChoiceIn("intent", ["cancel_account"]) & F.subscription_active & ScoreExpect("churn_risk", CHURN_E, CHURN_C),
             [pending("cancel_subscription"), R("propose_cancellation")],
             why="No retention offers: a clear, confident decision to leave is confirmed and honoured."),
        Rule("thinking of cancelling", ChoiceIn("intent", ["cancel_account"]), [R("ask_cancellation_reason")]),
        Rule("where is my order", ChoiceIn("intent", ["delivery_issue"]) & F.has_order, [R("provide_order_status")]),
        Rule("which order", ChoiceIn("intent", ["delivery_issue"]) & F.no_order, [R("ask_order_details")]),
        Rule("general", ChoiceIn("intent", ["general_question", "billing_dispute", "refund_request", "unauthorized_charge"]), [R("explain_charge_or_policy")]),
        Rule("fallback", Always(), [R("ask_clarify_issue")]),
    ]),
]

POLICY_A = Policy(
    key="A", name="Policy A — Guarded (card issuer, precision first)",
    summary=(
        "Policy A is written for a regulated card issuer whose worst outcome is an automated action taken wrongly. "
        "It hands off early and often: a threat, a request for a person, a third of the probability mass at 'angry' or "
        "above, an allegation that the company erred, or a claim by the assistant that the record contradicts all end the "
        "assistant's involvement. It executes only what the customer has explicitly accepted on the exact terms proposed, "
        "and even then only small refunds (at most $50) and reversible protections such as a card freeze. Anything larger, "
        "a request for more than the record shows, and every fraud finding once the card is handled, goes to a person. "
        "Investigation facts count at 0.7. Intent is recorded and acted on only above a confidence of 0.6; below that the assistant asks."
    ),
    sections=SECTIONS,
)
