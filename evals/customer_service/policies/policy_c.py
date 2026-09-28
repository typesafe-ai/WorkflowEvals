"""Choose a risk tier before applying its thresholds and action limits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..gates import Always, ChoiceIn, ChoiceTop, NodeRan, NoulGate, ScoreExpect, ScoreTail, forms_used
from . import facts as F
from .base import (CANCEL, CLEAR_PENDING, CLOSE, DISPUTE, FREEZE, MARK_RESOLVED, REDACT, REFUND, RETENTION, SET_INTENT, WAIVE_FEE,
                   Policy, R, Rule, Section, flag, handoff, pending, priority)

INTENT_CONF = 0.5
FRAUD = 0.5
CRED = 0.5
CALM_TAIL = 0.25       # Auto-close requires P(frustration >= 1) below this.
RESOLVED = 0.6
TERMS = 0.5

HIGH_INTENTS = ["unauthorized_charge", "account_access"]
MID_INTENTS = ["billing_dispute", "refund_request", "cancel_account"]
LOW_INTENTS = ["card_declined", "delivery_issue", "general_question"]


@dataclass(frozen=True)
class Tier:
    name: str
    handoff_level: int       # frustration level whose tail triggers a handoff
    handoff_tail: float
    human: float
    threat: float
    error: float
    false_claim: float
    consent: float
    fact: float
    auto_refund_max: float   # 0 disables automatic refunds.
    propose_refund_max: float
    urgency_level: int       # urgency level whose tail raises priority
    urgency_tail: float
    hardship_level: int
    hardship_tail: float
    churn_e: float
    churn_c: float


TIERS: Dict[str, Tier] = {
    "high": Tier("high", handoff_level=2, handoff_tail=0.4, human=0.5, threat=0.5, error=0.5, false_claim=0.5, consent=0.75, fact=0.75,
                 auto_refund_max=0.0, propose_refund_max=100.0, urgency_level=2, urgency_tail=0.5,
                 hardship_level=2, hardship_tail=0.5, churn_e=1.5, churn_c=0.5),
    "mid": Tier("mid", handoff_level=3, handoff_tail=0.5, human=0.6, threat=0.6, error=0.6, false_claim=0.6, consent=0.6, fact=0.65,
                auto_refund_max=30.0, propose_refund_max=200.0, urgency_level=3, urgency_tail=0.5,
                hardship_level=2, hardship_tail=0.6, churn_e=2.0, churn_c=0.6),
    "low": Tier("low", handoff_level=3, handoff_tail=0.7, human=0.7, threat=0.7, error=0.7, false_claim=0.7, consent=0.5, fact=0.6,
                auto_refund_max=60.0, propose_refund_max=300.0, urgency_level=3, urgency_tail=0.6,
                hardship_level=1, hardship_tail=0.7, churn_e=2.5, churn_c=0.7),
}
UNKNOWN = Tier("unknown", handoff_level=2, handoff_tail=0.5, human=0.5, threat=0.5, error=0.5, false_claim=0.5, consent=0.75, fact=0.75,
               auto_refund_max=0.0, propose_refund_max=100.0, urgency_level=2, urgency_tail=0.5,
               hardship_level=2, hardship_tail=0.5, churn_e=1.5, churn_c=0.5)

ELIGIBLE_REASONS = ["damaged_or_faulty", "never_arrived", "wrong_item", "duplicate_or_erroneous_charge"]
money_context = ChoiceIn("intent", ["refund_request", "billing_dispute"]) | ChoiceIn("desired_outcome", ["money_back"])

STAGE1 = [
    Rule("high tier", ChoiceTop("intent", INTENT_CONF) & (ChoiceIn("intent", HIGH_INTENTS) | NoulGate("reports_unauthorized", FRAUD)), [],
         why="Anything touching unauthorized activity or account takeover is high risk."),
    Rule("mid tier", ChoiceTop("intent", INTENT_CONF) & ChoiceIn("intent", MID_INTENTS), []),
    Rule("low tier", ChoiceTop("intent", INTENT_CONF) & ChoiceIn("intent", LOW_INTENTS), []),
    Rule("unknown", Always(), [], why="Intent confidence at or below 0.5 (after the two-way check, if one ran): the assistant asks, unless a safety rule fires first."),
]


def tier_of(ans, case) -> str:
    for r, name in zip(STAGE1, ["high", "mid", "low", "unknown"]):
        if r.when.holds(ans, case):
            return name
    return "unknown"


def consent_rules(t: Tier) -> List[Rule]:
    accepts = ChoiceTop("proposal_reply", t.consent, "accepts") & ~NoulGate("changes_terms", TERMS)
    declines = ChoiceTop("proposal_reply", t.consent, "declines")
    changed = ChoiceTop("proposal_reply", t.consent, "accepts") & NoulGate("changes_terms", TERMS)
    return [
        Rule("accepted on different terms", changed, [R("ask_desired_outcome")], why="Not consent; the proposal stays pending."),
        Rule("freeze accepted", F.pending_is("freeze_card") & accepts & F.card_active, [FREEZE, CLEAR_PENDING, handoff("security"), R("confirm_card_frozen")]),
        Rule("freeze declined", F.pending_is("freeze_card") & declines, [CLEAR_PENDING, handoff("security"), R("acknowledge_no_freeze")]),
        Rule("refund accepted", F.pending_is("issue_refund") & accepts & F.refund_eligible & F.refund_at_most(t.propose_refund_max), [REFUND, CLEAR_PENDING, MARK_RESOLVED, R("confirm_refund_issued")]),
        Rule("refund accepted, over the tier limit", F.pending_is("issue_refund") & accepts & F.refund_eligible, [CLEAR_PENDING, handoff("refunds"), R("explain_refund_under_review")]),
        Rule("refund declined", F.pending_is("issue_refund") & declines, [CLEAR_PENDING, R("ask_desired_outcome")]),
        Rule("cancellation accepted", F.pending_is("cancel_subscription") & accepts & F.subscription_active, [CANCEL, CLEAR_PENDING, MARK_RESOLVED, R("confirm_cancellation")]),
        Rule("cancellation declined", F.pending_is("cancel_subscription") & declines, [CLEAR_PENDING, R("closing_acknowledgement")]),
        Rule("dispute accepted", F.pending_is("open_dispute") & accepts & F.has_transaction, [DISPUTE, CLEAR_PENDING, R("confirm_dispute_opened")]),
        Rule("dispute declined", F.pending_is("open_dispute") & declines, [CLEAR_PENDING, R("ask_desired_outcome")]),
        Rule("hardship referral accepted", F.pending_is("hardship_referral") & accepts, [CLEAR_PENDING, handoff("hardship"), R("handoff_notice")]),
        Rule("hardship referral declined", F.pending_is("hardship_referral") & declines, [CLEAR_PENDING, R("explain_charge_or_policy")]),
    ]


def sections_for_tier(t: Tier) -> List[Section]:
    explained = NoulGate("recognizes_charge", t.fact) | NoulGate("authorized_person_made_it", t.fact)
    takeover = NoulGate("account_or_device_anomalies", t.fact)
    lost = ChoiceTop("card_possession", t.fact, "lost_or_stolen")
    in_hand = ChoiceTop("card_possession", t.fact, "has_it")
    unknown_possession = NodeRan("fraud") & ~lost & ~in_hand
    return [
        Section(f"[{t.name}] Safety", [
            Rule("credentials exposed", NoulGate("shares_credentials", CRED), [REDACT, R("warn_not_to_share_credentials")]),
            Rule("legal or regulatory threat", NoulGate("threat_legal_regulatory", t.threat), [handoff("compliance"), flag("legal_threat"), R("handoff_notice")]),
            Rule("chargeback or public threat", NoulGate("threat_chargeback_or_public", t.threat), [flag("public_complaint_threat")], stop=False),
        ]),
        Section(f"[{t.name}] What the assistant claimed, against the record", [
            Rule("claimed a refund the record does not show", NoulGate("claims_refund_done", t.false_claim) & F.refund_not_issued, [flag("agent_error"), handoff("supervisor"), R("apologize_and_correct")]),
            Rule("claimed the card was secured while it is active", NoulGate("claims_secured", t.false_claim) & F.card_active, [flag("agent_error"), handoff("security"), R("apologize_and_correct")]),
        ], intro="Read from the integrity node."),
        Section(f"[{t.name}] Pending proposal", consent_rules(t), intro="Read from the consent node."),
        Section(f"[{t.name}] People", [
            Rule("asks for a person", NoulGate("requests_human", t.human), [handoff("general"), R("handoff_notice")]),
            Rule("hostile", ScoreTail("frustration", 4, 0.5), [flag("abusive_customer"), handoff("supervisor"), R("handoff_notice")]),
            Rule("too frustrated for this tier", ScoreTail("frustration", t.handoff_level, t.handoff_tail), [handoff("supervisor"), R("handoff_notice")]),
            Rule("alleges an error", NoulGate("claims_agent_error", t.error), [flag("agent_error"), R("apologize_and_correct")], stop=False),
            Rule("urgent for this tier", ScoreTail("urgency", t.urgency_level, t.urgency_tail), [priority("urgent")], stop=False),
        ], mode="all"),
        Section(f"[{t.name}] Closing", [
            Rule("resolved and calm", NoulGate("issue_resolved", RESOLVED) & ScoreTail("frustration", 1, CALM_TAIL, "<") & F.no_pending, [MARK_RESOLVED, CLOSE, R("closing_acknowledgement")],
                 why="Auto-close only when almost no mass sits at 'mildly irritated' or above."),
            Rule("resolved but not calm", NoulGate("issue_resolved", RESOLVED) & F.no_pending, [MARK_RESOLVED, R("closing_acknowledgement")],
                 why="Resolved, but the ticket stays open for a follow-up."),
        ]),
        Section(f"[{t.name}] Intent", [Rule("record intent", Always(), [SET_INTENT], stop=False)]),
        Section(f"[{t.name}] Unauthorized activity: the investigation tree", [
            Rule("explained after all", explained, [R("explain_charge_or_policy")]),
            Rule("card lost or stolen", lost & F.card_active, [FREEZE, priority("urgent"), handoff("security"), R("confirm_card_frozen")],
                 why="A lost card is frozen without asking; the loss is the consent."),
            Rule("account takeover", takeover & F.card_active, [FREEZE, priority("urgent"), handoff("security"), R("account_recovery_steps")]),
            Rule("possession not stated", unknown_possession & F.card_active & NoulGate("reports_unauthorized", FRAUD), [R("ask_card_in_possession")]),
            Rule("card in hand, details do not match", in_hand & ~NoulGate("details_match_ledger", t.fact), [R("ask_charge_details")]),
            Rule("card in hand, charge on file", in_hand & F.card_active & F.has_transaction, [pending("open_dispute"), R("explain_dispute_process")]),
            Rule("card in hand, nothing on file", in_hand & F.no_transaction, [R("ask_charge_details")]),
            Rule("card already frozen", NodeRan("fraud") & NoulGate("reports_unauthorized", FRAUD) & F.card_frozen, [handoff("security"), R("handoff_notice")]),
        ], intro=f"Read from the fraud node. Facts count at {t.fact} in this tier."),
        Section(f"[{t.name}] Money back", [
            Rule("hardship", money_context & F.has_fee & ScoreTail("hardship", t.hardship_level, t.hardship_tail), [WAIVE_FEE, MARK_RESOLVED, R("confirm_refund_issued")]),
            Rule("already compensated", money_context & (F.refund_already_issued | NoulGate("already_compensated", 0.6)), [R("explain_charge_or_policy")]),
            Rule("asks for more than the record", money_context & F.refund_eligible & ChoiceTop("amount_vs_record", 0.5, "more"), [handoff("refunds"), R("explain_refund_under_review")]),
            Rule("no reason yet", money_context & F.refund_eligible & ChoiceIn("refund_reason", ["no_reason_given"]), [R("ask_refund_reason")]),
            Rule("automatic refund", money_context & F.refund_eligible & F.refund_at_most(t.auto_refund_max) & ChoiceIn("refund_reason", ELIGIBLE_REASONS) & ChoiceTop("refund_reason", 0.6), [REFUND, MARK_RESOLVED, R("confirm_refund_issued")]),
            Rule("propose a refund", money_context & F.refund_eligible & F.refund_at_most(t.propose_refund_max) & ChoiceIn("refund_reason", ELIGIBLE_REASONS), [pending("issue_refund"), R("propose_refund")]),
            Rule("refund needs a person", money_context & F.refund_eligible, [handoff("refunds"), R("explain_refund_under_review")]),
            Rule("not eligible", money_context & F.refund_ineligible, [R("explain_refund_not_eligible")]),
            Rule("nothing on file", money_context & F.no_refund_claim, [R("ask_charge_details")]),
        ], intro="Read from the money node."),
        Section(f"[{t.name}] Other issues", [
            Rule("locked out", ChoiceIn("intent", ["account_access"]), [R("account_recovery_steps")]),
            Rule("card declined", ChoiceIn("intent", ["card_declined"]), [R("explain_charge_or_policy")]),
            Rule("where is my order", ChoiceIn("intent", ["delivery_issue"]) & F.has_order, [R("provide_order_status")]),
            Rule("which order", ChoiceIn("intent", ["delivery_issue"]) & F.no_order, [R("ask_order_details")]),
            Rule("leaving", ChoiceIn("intent", ["cancel_account"]) & F.subscription_active & ScoreTail("churn_risk", 3, 0.6) & ~NoulGate("open_to_offer", 0.5), [pending("cancel_subscription"), R("propose_cancellation")]),
            Rule("at risk", ChoiceIn("intent", ["cancel_account"]) & ScoreExpect("churn_risk", t.churn_e, t.churn_c), [RETENTION, R("offer_retention_incentive")]),
            Rule("wants to cancel", ChoiceIn("intent", ["cancel_account"]), [R("ask_cancellation_reason")]),
            Rule("general", Always(), [R("explain_charge_or_policy")]),
        ]),
    ]


UNKNOWN_SECTIONS = [
    Section("[unknown] Safety", [
        Rule("credentials exposed", NoulGate("shares_credentials", CRED), [REDACT, R("warn_not_to_share_credentials")]),
        Rule("legal or regulatory threat", NoulGate("threat_legal_regulatory", 0.5), [handoff("compliance"), flag("legal_threat"), R("handoff_notice")]),
        Rule("hostile", ScoreTail("frustration", 4, 0.5), [flag("abusive_customer"), handoff("supervisor"), R("handoff_notice")]),
        Rule("asks for a person", NoulGate("requests_human", 0.5), [handoff("general"), R("handoff_notice")]),
    ]),
    Section("[unknown] Pending proposal", consent_rules(UNKNOWN), intro="Read from the consent node."),
    Section("[unknown] Closing", [
        Rule("resolved and calm", NoulGate("issue_resolved", RESOLVED) & ScoreTail("frustration", 1, CALM_TAIL, "<") & F.no_pending, [MARK_RESOLVED, CLOSE, R("closing_acknowledgement")]),
        Rule("resolved but not calm", NoulGate("issue_resolved", RESOLVED) & F.no_pending, [MARK_RESOLVED, R("closing_acknowledgement")]),
    ]),
    Section("[unknown] Ask", [
        Rule("frustrated and unclear", ScoreTail("frustration", 2, 0.5), [flag("ambiguous_intent"), R("ask_clarify_issue")],
             why="Unclear and already frustrated: flagged so a person can look at the transcript."),
        Rule("unclear", Always(), [R("ask_clarify_issue")]),
    ]),
]

TIER_SECTIONS = {name: sections_for_tier(t) for name, t in TIERS.items()}
TIER_SECTIONS["unknown"] = UNKNOWN_SECTIONS


class TieredPolicy(Policy):
    def sections_for(self, ans, case):
        return TIER_SECTIONS[tier_of(ans, case)]

    def forms(self):
        s = set()
        for r in STAGE1:
            s |= forms_used(r.when)
        for secs in TIER_SECTIONS.values():
            for sec in secs:
                for r in sec.rules:
                    s |= forms_used(r.when)
        return s


POLICY_C = TieredPolicy(
    key="C", name="Policy C — Tiered (risk tier first, then tier-specific thresholds)",
    summary=(
        "Policy C decides how careful to be before it decides what to do. Stage 1 places the conversation in a risk tier: "
        "high for unauthorized activity and account takeover, mid for money and cancellations, low for everything else, "
        "or unknown when intent confidence is at or below 0.5. Stage 2 runs one set of rule templates whose every number "
        "comes from the tier: the high tier hands off when 40% of the frustration mass sits at 'clearly frustrated' or "
        "above, establishes investigation facts only at 0.75, and never refunds without a yes; the low tier tolerates 70% "
        "at 'angry', accepts facts at 0.6, and refunds up to $60 automatically. A lost card is frozen without a proposal "
        "step in every tier. A claim by the assistant that the record contradicts goes to a person. A resolved "
        "conversation is auto-closed only when less than a quarter of the frustration mass sits above 'calm'."
    ),
    sections=STAGE1,
)
