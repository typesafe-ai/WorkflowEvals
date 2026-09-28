from __future__ import annotations

from typing import List, Optional

from core.question_types import Choice, Noul, Question, QuestionSet, Score

INTENTS = {
    "unauthorized_charge": "The customer reports a charge, withdrawal, or login they say they did not make or authorize.",
    "card_declined": "The customer's card or payment is being declined or not working when they try to use it.",
    "refund_request": "The customer wants money back for a purchase, order, or service they did receive or agreed to.",
    "billing_dispute": "The customer questions a fee, a double charge, a wrong amount, or a charge they recognize but consider incorrect.",
    "account_access": "The customer cannot log in, is locked out, or needs to reset credentials or recover the account.",
    "cancel_account": "The customer wants to cancel, close, or downgrade their account or subscription.",
    "delivery_issue": "The customer's order or replacement card has not arrived, is late, or was delivered wrongly.",
    "general_question": "A question about how something works, limits, features, or anything not covered above.",
}


TRIAGE: QuestionSet = {
    "intent": Choice(
        instructions="What is the customer's primary issue in this conversation, judged by the outcome they want above all else?",
        criteria=INTENTS,
    ),
    "desired_outcome": Choice(
        instructions="What resolution does the customer say they want?",
        criteria={
            "money_back": "A refund, reversal, credit, or fee waiver.",
            "replacement_or_fix": "A replacement item, a new card, a repair, or the service made to work.",
            "explanation_only": "To understand what happened or how something works; no money or item requested.",
            "speak_to_human": "To talk to a person, a manager, or a specific department.",
            "not_stated": "The customer has not said what outcome they want.",
        },
    ),
    "frustration": Score(
        instructions="How frustrated is the customer in their latest message, in the context of the conversation?",
        criteria=[
            "Calm and neutral; no sign of irritation.",
            "Mildly irritated; a hint of impatience or disappointment but polite.",
            "Clearly frustrated; complains about the situation or the service, expresses annoyance directly.",
            "Angry; strong language, repeated complaints, demands, or expresses distrust of the company.",
            "Hostile or abusive; insults, threats toward the assistant, profanity directed at the company or agent.",
        ],
    ),
    "urgency": Score(
        instructions="How time-critical is the customer's situation, as they describe it?",
        criteria=[
            "No time pressure mentioned.",
            "Some inconvenience; they would like it handled soon.",
            "Pressing: a deadline, travel, a payment due, or repeated failed attempts are mentioned.",
            "Emergency: stranded, locked out with no alternative, unable to pay for essentials, or ongoing unauthorized use right now.",
        ],
    ),
    "reports_unauthorized": Noul(
        instructions="Does the customer report a charge, withdrawal, login, or account change that they say they did not make or authorize?",
        criteria={
            "true": "The customer states or strongly implies someone else made a charge or accessed the account.",
            "false": "The customer recognizes the activity, disputes only its amount or fairness, or reports nothing unauthorized.",
        },
    ),
    "threat_legal_regulatory": Noul(
        instructions="Does the customer themselves threaten or announce legal action, a lawyer, a regulator, an ombudsman, or a formal complaint to an authority?",
        criteria={
            "true": "The customer says they will, or are about to, involve a lawyer, court, regulator, attorney general, or similar body.",
            "false": "No such threat, or it is hypothetical, quoted from someone else, or refers to the past.",
        },
    ),
    "threat_chargeback_or_public": Noul(
        instructions="Does the customer threaten a bank chargeback, a public review, social media exposure, or press attention?",
        criteria={
            "true": "The customer says they will dispute the charge with their bank, post publicly, or contact the media.",
            "false": "No such threat, or it is hypothetical, quoted, or in the past.",
        },
    ),
    "requests_human": Noul(
        instructions="Does the customer ask to speak with a human, a real person, a manager, or a supervisor?",
        criteria={
            "true": "The customer explicitly asks for a person, a manager, or to be transferred away from the bot.",
            "false": "The customer does not ask for a human.",
        },
    ),
    "shares_credentials": Noul(
        instructions="Does the customer's message contain a full card number, CVV, PIN, password, or Social Security number written out?",
        criteria={
            "true": "A complete sensitive secret appears in the text.",
            "false": "Only a last-four, a partial, a masked value, or no secret appears.",
        },
    ),
    "issue_resolved": Noul(
        instructions="Does the customer indicate that their issue is now resolved or that they no longer need help with it?",
        criteria={
            "true": "The customer says it is sorted, they found the answer, never mind, or thanks the assistant and signals they are done.",
            "false": "The customer still has an open request, asks a new question, or is waiting on something.",
        },
    ),
    "claims_agent_error": Noul(
        instructions="Does the customer say the assistant or company gave them wrong information, did not do what it promised, or made a mistake in this or a previous contact?",
        criteria={
            "true": "The customer alleges a specific error, broken promise, or wrong answer by the company or assistant.",
            "false": "The customer does not allege an error by the company.",
        },
    ),
}


CONSENT: QuestionSet = {
    "proposal_reply": Choice(
        instructions="How does the customer reply to the assistant's proposed action?",
        criteria={
            "accepts": "The customer clearly says yes to the proposed action as stated.",
            "declines": "The customer clearly says no, or asks the assistant not to do it.",
            "unclear_or_conditional": "The customer asks a question, adds a condition, changes the terms, or gives no clear yes or no.",
        },
    ),
    "changes_terms": Noul(
        instructions="Does the customer's reply ask for different terms than proposed, such as a different amount, a different action, or an extra condition?",
        criteria={
            "true": "The reply names a different amount, adds a condition, or swaps the action for another one.",
            "false": "The reply keeps the proposal as stated, or does not engage with its terms.",
        },
    ),
}


FRAUD: QuestionSet = {
    "card_possession": Choice(
        instructions="Does the customer say whether they still physically have their card?",
        criteria={
            "has_it": "The customer says the card is with them, in their wallet, or in their possession.",
            "lost_or_stolen": "The customer says the card is lost, stolen, missing, or they cannot find it.",
            "not_stated": "The customer does not say either way.",
        },
    ),
    "recognizes_charge": Noul(
        instructions="Does the customer now recognize the charge as their own, or say it has been explained?",
        criteria={
            "true": "The customer says it was theirs after all, remembers the purchase, or accepts the explanation.",
            "false": "The customer still does not recognize the charge, or has not said.",
        },
    ),
    "authorized_person_made_it": Noul(
        instructions="Does the customer say or accept that a family member, partner, or other person with access to the card made the charge?",
        criteria={
            "true": "The customer identifies someone with access who made it, or agrees that is what happened.",
            "false": "No such person is identified, or the customer denies anyone else has access.",
        },
    ),
    "other_unfamiliar_activity": Noul(
        instructions="Does the customer report other charges, withdrawals, or logins they do not recognize, beyond the one under discussion?",
        criteria={
            "true": "The customer mentions additional unrecognized activity.",
            "false": "Only the single charge under discussion is mentioned.",
        },
    ),
    "account_or_device_anomalies": Noul(
        instructions="Does the customer describe account or device anomalies such as unrequested login codes, a changed email or phone number, a password reset they did not request, or an unknown device?",
        criteria={
            "true": "The customer describes at least one such anomaly.",
            "false": "No account or device anomaly is described.",
        },
    ),
    "details_match_ledger": Noul(
        instructions="Do the merchant, amount, and date the customer describes match the ledger entry supplied?",
        criteria={
            "true": "The customer's description matches the ledger entry.",
            "false": "The customer's description differs from the ledger entry, or no ledger entry is supplied.",
        },
    ),
}


MONEY: QuestionSet = {
    "refund_reason": Choice(
        instructions="What reason does the customer give for wanting money back or a charge reversed?",
        criteria={
            "damaged_or_faulty": "The item arrived damaged, broken, defective, or stopped working.",
            "never_arrived": "The item or service was paid for but never received.",
            "wrong_item": "The wrong item, size, color, or quantity was delivered.",
            "duplicate_or_erroneous_charge": "They were charged twice, charged the wrong amount, or charged for something they did not order.",
            "changed_mind": "They no longer want the item or service, found it cheaper, or bought it by mistake; the item itself is fine.",
            "no_reason_given": "The customer does not give a reason, or is not asking for money back at all.",
        },
    ),
    "amount_vs_record": Choice(
        instructions="How does the amount the customer asks for compare with the amount on file?",
        criteria={
            "same": "The customer asks for the amount on file, or does not name an amount and refers to the charge or fee as a whole.",
            "more": "The customer asks for more than the amount on file, for example adding shipping, other charges, or compensation.",
            "less": "The customer asks for part of the amount on file.",
            "not_asking_for_money": "The customer is not asking for money back at all.",
        },
    ),
    "already_compensated": Noul(
        instructions="Does the customer say they have already received a refund, credit, or waiver for this?",
        criteria={
            "true": "The customer mentions money or credit already received for this issue.",
            "false": "The customer does not mention having received anything.",
        },
    ),
    "offers_evidence": Noul(
        instructions="Does the customer offer or mention evidence such as photos, receipts, tracking, bank statements, or a confirmation number?",
        criteria={
            "true": "The customer offers or references concrete evidence.",
            "false": "No evidence is offered or referenced.",
        },
    ),
    "hardship": Score(
        instructions="Does the customer describe financial hardship, and how severe?",
        criteria=[
            "No mention of financial difficulty.",
            "Mentions money is tight, or that the amount matters to them.",
            "Says the charge or fee causes real difficulty: overdraft, missed bill, cannot afford it this month.",
            "Describes being unable to cover essentials such as rent, food, or medication, or a crisis such as job loss.",
        ],
    ),
}


RETENTION: QuestionSet = {
    "churn_risk": Score(
        instructions="How likely is the customer to leave the company, judging by what they say?",
        criteria=[
            "No sign of leaving; the customer just wants the issue handled.",
            "Hints at dissatisfaction with the company overall, or compares it unfavorably to others.",
            "Says they are considering leaving, switching, or cancelling if this is not resolved.",
            "States they are cancelling, closing the account, or have already decided to leave.",
        ],
    ),
    "cancellation_reason": Choice(
        instructions="What reason does the customer give for wanting to cancel or leave?",
        criteria={
            "price": "The price, a price increase, or a fee.",
            "not_using": "They do not use or need the service.",
            "competitor": "A better offer or experience elsewhere.",
            "service_failure": "A problem, error, or bad experience with the company.",
            "not_stated": "No reason is given, or the customer is not cancelling.",
        },
    ),
    "open_to_offer": Noul(
        instructions="Does the customer signal they would consider staying if something were offered or fixed?",
        criteria={
            "true": "The customer conditions leaving on a fix, a discount, or an offer, or asks what can be done.",
            "false": "The customer has decided, or does not engage with staying.",
        },
    ),
}


def intent_pair(a: str, b: str) -> QuestionSet:
    return {
        "intent_pair": Choice(
            instructions="Which of these two better describes the customer's primary issue, judged by the outcome they want above all else?",
            criteria={a: INTENTS[a], b: INTENTS[b]},
        )
    }


def integrity_questions(message_numbers: List[int]) -> QuestionSet:
    qs: QuestionSet = {}
    for k in message_numbers:
        qs[f"refund_done_msg_{k}"] = Noul(
            instructions=f"Does assistant message number {k}, on its own, state that a refund, credit, or waiver has already been issued or completed?",
            criteria={
                "true": "The message says the money has been refunded, credited, or waived, in the past tense or as done.",
                "false": "The message proposes, promises, or conditions a future refund, or says nothing about one.",
            },
        )
        qs[f"secured_msg_{k}"] = Noul(
            instructions=f"Does assistant message number {k}, on its own, state that the card or account has already been frozen, blocked, or secured?",
            criteria={
                "true": "The message says the card or account is now frozen, blocked, locked, or secured.",
                "false": "The message proposes a freeze, says nothing about it, or describes it as a future step.",
            },
        )
    return qs


ALL_STATIC: QuestionSet = {**TRIAGE, **CONSENT, **FRAUD, **MONEY, **RETENTION}
QUESTIONS = ALL_STATIC  # Dynamic intent-pair and integrity IDs are resolved at runtime.


def spec_of(key: str, qs: Optional[QuestionSet] = None) -> Question:
    if qs and key in qs:
        return qs[key]
    if key in ALL_STATIC:
        return ALL_STATIC[key]
    if key == "intent_pair":
        return Choice(instructions="", criteria={})
    if key.startswith(("refund_done_msg_", "secured_msg_")):
        return Noul(instructions="", criteria={"true": "", "false": ""})
    raise KeyError(key)
