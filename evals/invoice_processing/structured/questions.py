# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.question_types import Choice, Noul, NoulCriteria, QuestionSet

from .facts import RETAINAGE_LEVELS, ExactFacts
from .findings import (
    DISCOUNT_DAY_LEVELS,
)
from .state import Packet

MAX_LINE_QUESTIONS = 12


@dataclass(frozen=True)
class SemanticQuestion:
    id: str
    question: Noul | Choice
    axis: str
    applies: Callable[[ExactFacts], bool] = lambda _facts: True
    skip_reason: str = ""


UNSURE = (
    "UNSURE: none of the other options is more likely than not; the packet genuinely supports two of them. Not a "
    "way to defer a decision you could make."
)


def _choice(instructions: str, criteria: dict[str, str], unsure: bool = True) -> Choice:
    crit = dict(criteria)
    if unsure:
        crit["unsure"] = UNSURE
    return Choice(instructions=instructions, criteria=crit)


PRICE_BASIS_TEXT: dict[str, str] = {
    "total_fee": (
        "The contract or SOW text in the packet states ONE TOTAL FEE for the whole engagement or statement of work "
        "across all its invoices (a 'Contract Sum', 'total fixed fee', 'total fees shall not exceed' for the agreed "
        "scope), and that total fee is the price basis for what this invoice bills. Not this: a per-period fee, a PO "
        "ceiling or budget bucket above the fee, a sub-limit, a milestone amount, a penalty, deposit or insurance figure."
    ),
    "period_fee": (
        "The contract or SOW text states a RECURRING FEE PER PERIOD (a monthly, quarterly or annual retainer, "
        "subscription or service fee) and that fee is the price basis for what this invoice bills. Not this: a total "
        "fee for the whole engagement, a cap, a sub-limit such as a monthly pass-through maximum, or per-unit rates."
    ),
    "unit_rates": (
        "The contract or SOW text states RATES OR FIXED AMOUNTS for the billed items: per hour, day, unit, seat, "
        "licence or mile, or a fixed amount per milestone or deliverable, and those are the price basis for what "
        "this invoice bills, with the figures stated in the text. Purchase-order line prices do not count (code "
        "compares them); a rate card the text names but does not state is not a basis."
    ),
    "cap_only": (
        "The ONLY price-like FIGURE the contract or SOW text states for what this invoice bills is a maximum, "
        "not-to-exceed or budget bucket that is not itself the fee for the work (a PO ceiling, 'charges shall not "
        "exceed USD X without written authorization'). YES only if the cap FIGURE itself is written in the contract or "
        "SOW text; a not-to-exceed the text mentions without a number ('shall not exceed the amount stated on the "
        "PO') is not a stated figure, and then the basis is none, NO here. A fee or rate card the text names but "
        "does not state ('rates per Annex A') is likewise not a figure. If the text states a fee, rate or milestone "
        "amount for the work, that figure is the basis and this is NO."
    ),
}

QUESTION_BANK: list[SemanticQuestion] = [
    SemanticQuestion(
        id="statement_not_invoice",
        axis="document",
        question=Noul(
            instructions=(
                "The document in the packet is NOT an invoice: it is a statement of account, a quote, a pro-forma, "
                "an estimate, a reminder or a remittance summary. A statement that says an invoice is attached is "
                "still a statement. A tax invoice or a credit note IS an invoice-type document."
            ),
        ),
    ),
    SemanticQuestion(
        id="different_entity",
        axis="authorization",
        question=Noul(
            instructions=(
                "The invoice appears to belong to a different project, statement of work, "
                "business unit, or legal entity than the PO it is matched to. A bill-to department or cost centre "
                "that differs from the PO's, or a bill-to entity or vendor mismatch the system has already found, is NOT this on "
                "its own: answer YES only where the invoice's CONTENT places it in a different project, SOW, unit or entity."
            ),
        ),
    ),
    SemanticQuestion(
        id="too_ambiguous",
        axis="information_quality",
        question=Noul(
            instructions=(
                "At least one substantive line of this invoice is described so generically that its own text "
                "does not say what was bought, for which period, or under which authorization: "
                '"Professional services", "Consulting services", "Services rendered", "Work performed", '
                '"Project work per agreement", "Professional fees". Judge the line text itself, not what the '
                "PO or contract would let you guess. A line that names a deliverable, item, part or work code is "
                "specific even if brief; a period, quantity or rate alone does not rescue a line that never says "
                "what was bought. A code or abbreviation that any document in the packet decodes is specific; one "
                "nothing in the packet decodes is vague; a work-order, ticket or project code counts as specific only where the "
                "packet ties it to a described item, service or deliverable. Naming only the project or engagement the work "
                "relates to does not cure a line that never says what was bought. Tax, freight and expense lines do not count."
            ),
        ),
    ),
    SemanticQuestion(
        id="unexplained_charges",
        axis="information_quality",
        question=Noul(
            instructions=(
                "The invoice's POSITIVE adjustments, that is fees, surcharges, handling or 'other' charges added below the "
                "lines, or a 'discount', 'rebate' or 'credit' LINE that is shown as a positive amount, trace to no line "
                "item, no contract clause, no PO term, and not to how this vendor has billed before. A credit or "
                "discount that actually reduces what is owed never makes this true. If the invoice has no positive "
                "adjustment of either kind, NO. An adjustment that duplicates a line is not thereby explained: it is also "
                "unexplained unless a contract clause or this vendor's prior billing provides for the charge."
            ),
        ),
    ),
    SemanticQuestion(
        id="adjustment_duplicates_line",
        axis="information_quality",
        question=Noul(
            instructions=(
                "A positive summary adjustment charges for something a line item on this invoice already "
                "charges: freight or handling billed as a line AND again as an adjustment, tax added below the "
                "lines when a line already carries it, a fee that repeats a line. Answer NO when the adjustment "
                "is a different charge from every line."
            ),
        ),
        applies=lambda f: f.summary_adjustments_total > 0,
        skip_reason="no positive summary adjustments",
    ),
    SemanticQuestion(
        id="tax_two_rates",
        axis="information_quality",
        question=Noul(
            instructions=(
                "Tax on this invoice is charged at two DIFFERENT rates (for example a 20% tax line inside the "
                "subtotal and 8.25% tax in the summary), or a tax line states a rate that does not match the "
                "summary tax. Tax charged once at one rate, or a legitimately mixed-rate supply the lines "
                "explain, is NO. Do not judge whether tax is charged twice at the same rate; code compares the "
                "amounts. A rate is read only from a percentage the line or summary states; a tax line stating no rate does not "
                "by itself establish two rates."
            ),
        ),
        applies=lambda f: f.summary_tax > 0 or len(f.lines) > 0,
        skip_reason="no tax on the invoice",
    ),
    SemanticQuestion(
        id="unusual_urgency",
        axis="fraud",
        question=Noul(
            instructions=(
                "A vendor-side message shows the PRESSURE PATTERN: a deadline measured in hours or a few days "
                "COMBINED WITH one of (i) an instruction not to use the phone number or contact on file, "
                "(ii) a claim that we already verified or confirmed something we have no record of, or (iii) a "
                "threat to withhold deliverables, certificates, releases or service, or a penalty, late fee or "
                "'account frozen' story tied to that deadline, beyond what the contract itself provides. "
                "Normal notices never count: '60 days overdue, service suspends per contract', polite chasers, "
                "'please prioritise', a reminder that a due date has passed. A vendor's claim that its own bank account is frozen, "
                "blocked or under review, offered with new payment details and a short deadline, IS the pattern."
            ),
        ),
        applies=lambda f: f.has_external_communications,
        skip_reason="no external (vendor-side) communications in the packet",
    ),
    SemanticQuestion(
        id="bank_change_claimed_in_comms",
        axis="fraud",
        question=Noul(
            instructions=(
                "A vendor-side message says the vendor's payment or bank details have changed, gives account "
                "details to pay to that differ from the vendor master (even while calling them unchanged), or "
                "claims a bank change was already confirmed or verified with us. Merely naming a bank in passing, "
                "with no change or account details, does NOT count; but a bank named AS the payment destination that differs from "
                "the vendor master's bank IS a claim even without an account number."
            ),
        ),
        applies=lambda f: f.has_external_communications,
        skip_reason="no external (vendor-side) communications in the packet",
    ),
    SemanticQuestion(
        id="price_basis",
        axis="price",
        question=_choice(
            "Which ONE of these describes the price the contract or SOW text states for the items this invoice bills? "
            "A basis needs a figure written in the text; a fee or rate card the text names but does not state is no basis.",
            {
                **PRICE_BASIS_TEXT,
                "none": (
                    "The contract or SOW text states no price figure for the items invoiced: no fee, rate, milestone "
                    "amount or not-to-exceed figure written in the text (a not-to-exceed that appears only in the "
                    "purchase order's own text, or a fee stated only for other entities, sites or streams, is none)."
                ),
            },
        ),
        applies=lambda f: f.contract_text_present,
        skip_reason="no contract text to read",
    ),
    SemanticQuestion(
        id="billed_above_basis",
        axis="price",
        question=Noul(
            instructions=(
                "Take the price basis the contract or SOW text states for what this invoice bills, and compare like "
                "with like: ex-tax, net of any escalation or indexation the contract itself permits. If the basis is a "
                "TOTAL fee for the engagement: everything billed so far including this invoice (exact_facts.cumulative_billed "
                "is the structured figure; add paid priors if it is missing) exceeds it. If it is a fee PER PERIOD: this "
                "invoice's substantive lines together bill more than the fee for the period. If it is RATES or fixed "
                "milestone amounts: a line, or a milestone or period billed across several lines taken together, is "
                "billed higher than its stated rate or amount. If it is only a CAP: cumulative billing exceeds a cap that "
                "appears only in the prose; a cap that is also the PO's structured maximum or the contract's structured "
                "maximum is compared by code, and when the prose figure conflicts with the structured one the structured "
                "one governs, so NO. Billing at or below the basis is NO. If the text states no price for what is billed, NO. If "
                "the invoice currency differs from the currency of the stated basis, NO (no comparison is made)."
            ),
            criteria=NoulCriteria(
                true="The invoice, measured the way the stated basis requires, is above the stated figure.",
                false="At or below the figure, an escalation the contract permits, a tax-inclusive total compared with an ex-tax fee, or no figure stated.",
            ),
        ),
        applies=lambda f: f.contract_text_present,
        skip_reason="no contract text to read",
    ),
    SemanticQuestion(
        id="lines_restate_po_pricing",
        axis="price",
        question=Noul(
            instructions=(
                "The invoice bills the PO's priced items split or re-labelled across lines (for example a bundled "
                "PO price billed as a base line plus a separate component line, or one PO item spread over two "
                "lines), so that a line-by-line unit-price comparison misleads while the same goods or services "
                "are billed at what the PO prices imply. Answer yes only when the PO line description itself "
                "shows the split is the same thing priced differently."
            ),
        ),
        applies=lambda f: bool(f.po_lines),
        skip_reason="no PO lines to compare against",
    ),
]

QUESTIONS_BY_ID: dict[str, SemanticQuestion] = {q.id: q for q in QUESTION_BANK}


@dataclass(frozen=True)
class LineQuestion:
    """Per-line questions use IDs of the form line_{index}_{kind}."""

    kind: str
    axis: str
    template: Callable[[int, str], str]  # (line number, short description) -> question text


LINE_KIND_QUESTIONS: dict[str, Callable[[int, str], str]] = {
    "work_or_goods": lambda n, short: (
        f'Invoice line {n} ("{short}") bills WORK OR GOODS: a service, deliverable, milestone, period of service, '
        "labour hours, usage, licences or physical items. Not tax, not a freight/handling/surcharge line, not a "
        "credit, discount, rebate, retainage or expense line."
    ),
    "tax": lambda n, short: (
        f'Invoice line {n} ("{short}") IS a tax amount (VAT, GST, IGST, sales tax, HST) charged on the other lines, '
        "whatever it is called. Work ABOUT tax (tax advisory, VAT return preparation, a tax compliance calendar) "
        "is work, not tax."
    ),
    "charge_on_top": lambda n, short: (
        f'Invoice line {n} ("{short}") is a CHARGE ON TOP of the work rather than work itself: freight, shipping, '
        "handling, courier, postage, a surcharge, an expediting, admin, processing or minimum charge, or a commission "
        "or mark-up computed as a percentage of other lines. Freight booking or dispatch paperwork accompanying goods "
        "is a charge on top; a line that itself delivers a service or goods is work."
    ),
    "credit_or_discount": lambda n, short: (
        f'Invoice line {n} ("{short}") is a CREDIT, discount, rebate or credit-memo line, whether shown negative '
        "or (wrongly) positive. It reduces or adjusts what is owed rather than billing work."
    ),
    "expense_or_retainage": lambda n, short: (
        f'Invoice line {n} ("{short}") is an EXPENSE pass-through (travel, per diem, mileage reimbursement) or a '
        "retainage / retention release line. Billed usage of a service (fleet mileage as the service itself) is "
        "work, not an expense."
    ),
}

WRONG_ENTITY = (
    "If the invoice's bill-to entity or vendor does not match its purchase order, the review stops before scope "
    "matters: answer YES to priced_unit and NO to every other scope option."
)
LINE_SCOPE_QUESTIONS: dict[str, Callable[[int, str], str]] = {
    "priced_unit": lambda n, short: (
        f'Invoice line {n} ("{short}") IS what the purchase order, SOW or contract authorizes: the priced item, '
        "milestone, period or deliverable itself, or the ordinary work an umbrella or retainer authorization "
        "describes. A line of a covered KIND of work counts even when its description is brief or generic "
        "(vagueness is a separate question). NOT this: a part of something another line already bills, an add-on "
        "or extra beyond the priced units (approved or not), or work the authorization never describes. "
        + WRONG_ENTITY
    ),
    "component_also_billed": lambda n, short: (
        f'Invoice line {n} ("{short}") bills a PART of something ANOTHER line on this same invoice already bills in '
        "full, so the invoice charges that work twice: a flat monthly fee plus a prorated or catch-up charge for the "
        "same window, the same milestone under two labels, a component billed separately when another line's price "
        "already includes it by its own description, or hourly lines for work the retainer already covers (the "
        "retainer's scope as the contract or PO gives it, or as the line itself says it is included, not only the "
        "retainer line's printed words). NO for different roles, phases, sites, deliverables or periods of one project, for "
        "hourly lines for work OUTSIDE the retainer's scope, and for lines that merely share a generic description. "
        + WRONG_ENTITY
    ),
    "extra_approved": lambda n, short: (
        f'Invoice line {n} ("{short}") is work BEYOND the priced units (an hourly or daily add-on on a fixed-fee, '
        "retainer or milestone contract; a change; extra units) AND the packet shows it was approved in writing "
        "before or when it was done: a change order, a PO line for it, an email or approval comment from the "
        "designated approver naming it, or a verbal emergency approval that the contract itself permits. NO if the "
        "line is the priced work itself, or if no such written approval is in the packet. " + WRONG_ENTITY
    ),
    "extra_unapproved": lambda n, short: (
        f'Invoice line {n} ("{short}") is work of a covered KIND beyond the priced units (an hourly or daily add-on '
        "on a fixed-fee, retainer or milestone contract; extra units; a change) WITHOUT written approval in the "
        "packet: a verbal request the contract does not provide for, a later tracker mention, an owner who says they "
        "asked informally, a pre-approval the packet never shows. NO if the line is the priced work itself, if written "
        "approval is in the packet, or if the work is of a kind the authorization never describes at all. "
        + WRONG_ENTITY
    ),
    "not_ours": lambda n, short: (
        f'Invoice line {n} ("{short}") is NOT the work the authorization describes at all: a different service, '
        "deliverable or project, a kind of work the contract or PO scope text never mentions, or an item the PO or "
        "owner explicitly excludes. NO for an add-on of a covered kind that merely lacks its own approval, and for a "
        "line whose wording is too vague to place. " + WRONG_ENTITY
    ),
}

LINE_COMPLETION_QUESTIONS: dict[str, Callable[[int, str], str]] = {
    "confirmed": lambda n, short: (
        f"OUR OWN records (a goods receipt, receiving confirmation, acceptance record, owner tracker note, owner "
        f'comment or an unqualified owner approval) say the work or goods on invoice line {n} ("{short}") were '
        "delivered or performed as billed for the billed period or milestone, or delivered with only the formal "
        "sign-off still pending. A record that describes the deliverable as a whole confirms every line of it. "
        "When our records disagree, the acceptance record, then the goods receipt, governs the count and an owner "
        "note governs degree and quality. NO if our record says less or nothing was done, if only the vendor's own "
        "documents describe it, or if nothing describes it."
    ),
    "less": lambda n, short: (
        f'OUR OWN record says LESS was done or delivered on invoice line {n} ("{short}") than the line bills: a '
        "lower percent complete, fewer units or a smaller headcount, components not fitted, deliverables the owner "
        "says are missing, receiving short (an acceptance record, or a goods receipt when there is no acceptance "
        "record). Only the buyer's own words count. When our records disagree, the acceptance record, then the goods "
        "receipt, governs the count and an owner note governs degree and quality; a goods receipt plainly wrong for "
        "lines other records verify is disregarded. NO if our record says nothing was done at all, if the "
        "disagreement is about whether the work was AUTHORISED (scope) or about price, or if the only conflicting "
        "document is the vendor's."
    ),
    "not_done": lambda n, short: (
        f"Our own records, or the vendor's own admission in the thread, say the work or goods on invoice line {n} "
        f'("{short}") have NOT HAPPENED AT ALL: not started, nothing delivered, milestone not reached. Partly done, '
        "done differently, fewer units or a lower percent complete is NO (that is 'less'). A milestone billed before "
        "its acceptance with no record saying the work has not happened is NO."
    ),
    "vendor_only": lambda n, short: (
        f'The ONLY evidence describing the work or goods on invoice line {n} ("{short}") is the vendor\'s own: its '
        "service log, its report, its delivery note, its timesheet. Nothing from our side (receipt, receiving, "
        "acceptance, owner note or an approval without comment) describes this line's work as delivered, short or "
        "not done. NO only if such a record of ours exists (a note that our review is still pending is not one), "
        "or if nothing at all describes it."
    ),
    "none": lambda n, short: (
        f'NOTHING in the packet describes the work or goods on invoice line {n} ("{short}") as delivered or '
        "performed for the billed period: no record of ours and no vendor document either. Unqualified owner "
        "approval of the invoice counts as our record (then NO)."
    ),
}

LINE_QUESTIONS: tuple[LineQuestion, ...] = (
    LineQuestion(
        "rebilled",
        "line_duplicate",
        lambda n, short: (
            f'Invoice line {n} ("{short}") charges for work, a period, a milestone or a deliverable (or a component '
            "of one) that a PRIOR invoice from this vendor (see prior_invoices) already billed and we PAID or have "
            "PENDING: the same shipment, the same application or milestone, the same service weeks. Wording, amount "
            "or invoice number may differ. Answer NO for a recurring service billing a NEW period (a later month, a "
            "later application number) however alike the prior invoices look, for a genuinely later phase or "
            "shipment, for a re-bill of a prior invoice that was reversed or credited, and when no prior invoice "
            "covers this work. Answer NO where the prior invoice bills this invoice's obligation AS A WHOLE (the same "
            "milestone, application, shipment or period that the whole invoice bills, or a re-issue of the same invoice): "
            "that is the prior-relation question, not a re-billed line; re-billed is for one old line slipped into an "
            "otherwise new invoice. YES for a catch-up or cumulative-adjustment charge for earlier months whose retainer a paid "
            "prior invoice already billed. If you cannot point to the specific prior invoice that already billed it, answer NO."
        ),
    ),
    LineQuestion(
        "owner_declined",
        "line_owner",
        lambda n, short: (
            f'The project owner or approver has declined invoice line {n} ("{short}") BY NAME in the packet: an '
            "approval comment, owner note or email saying this specific item was not commissioned, not requested, "
            'not authorised, or is excluded ("approved except for X", "we did not order the Y"); rejecting a named '
            "line because it was never authorised declines it, and so does an approval that accepts the invoice except "
            "this named line, whatever reason it gives. A general objection to the invoice, a pending approval, a "
            "question about quantity or completion, an owner who says they asked for the work informally, a note that "
            "the work needs a separate PO or change order or belongs to another programme without refusing to pay it, "
            "or silence about this line is NOT a declined line; answer NO. An objection to a named line only because its work "
            "is incomplete or unverified does not decline it."
        ),
    ),
    LineQuestion(
        "unexplained_fee",
        "line_charges",
        lambda n, short: (
            f'Invoice line {n} ("{short}") is an ADD-ON charge rather than the work or goods themselves: a '
            "fee, surcharge, handling, admin, expediting, minimum-charge or adjustment line (a 'coordination fee', "
            "not coordination labour on site) that sits on top of the priced work and traces to no PO line, no "
            "contract clause, and not to how this vendor has billed before (see prior invoices). A line billing "
            'services or goods is NOT a fee however vague its wording ("Consulting services", "Professional fees", '
            '"Work performed"); vagueness is a separate question. A credit, discount or rebate line, even shown as a '
            "positive amount, is never a fee (it is an adjustment); a tax or expense line is not a fee. Fees the "
            "contract or PO text provides for, and charges a PO line prices, are explained; scope prose is not a PO line. A "
            "commission or mark-up computed as a percentage of other lines is a charge on top, explained where the contract "
            "provides for it."
        ),
    ),
    LineQuestion(
        "rate_differs",
        "line_price",
        lambda n, short: (
            f'The text of invoice line {n} ("{short}") itself states a per-unit rate (per hour, day, unit, mile, seat, '
            "licence or month, e.g. '136 hrs @ $210/hr', '1,408 miles @ £0.42/mi'), and the price the line is "
            "actually billed at (its amount divided by its quantity, or its printed unit price) is HIGHER than "
            "that stated rate. A line whose text states no rate, or that is billed at or below the stated rate, "
            "is NO. Do the one division needed; nothing else."
        ),
    ),
)
LINE_QUESTION_KINDS: tuple[str, ...] = tuple(q.kind for q in LINE_QUESTIONS) + (
    "kind",
    "scope",
    "completion",
    "po",
    "evidence",
)
CANDIDATE_QUESTION_KINDS: tuple[str, ...] = (
    "prior:relation",
    "approval:reading",
    "sender:identity",
    "retainage:level",
    "discount:days",
)

PRIOR_RELATION_TEXT: dict[str, Callable[[str], str]] = {
    "same_obligation": lambda who: (
        f"Prior invoice {who} bills the SAME economic obligation as this invoice: the same milestone, application, "
        "shipment, period or deliverable, however the two are worded, numbered or priced; a re-issue of that prior "
        "under a new number is this too. NO for an earlier period of a recurring service, an earlier phase or "
        "shipment, or different work."
    ),
    "earlier_period_or_phase": lambda who: (
        f"Prior invoice {who} bills ANOTHER period of the same recurring service, or another phase, milestone or "
        "shipment of the same arrangement, whether before or after this one, and this invoice bills a DIFFERENT one. "
        "The amount does not decide this: a prorated, larger or smaller bill for a new period is still a new period. "
        "NO if the two bill the same obligation, or if the prior is unrelated work under a different arrangement."
    ),
    "unrelated": lambda who: (
        f"Prior invoice {who} bills work that is UNRELATED to this invoice: a different project, service, order or "
        "arrangement, neither the same obligation nor an earlier period or phase of the same one."
    ),
}

APPROVAL_READING_TEXT: dict[str, Callable[[str], str]] = {
    "approves_as_billed": lambda who: (
        f"The approval comment {who} ACCEPTS the invoice as billed: a routine remark ('OK to pay', 'looks good', a "
        "recode or cost-centre note) with no line carved out, no defect named, and nothing withheld."
    ),
    "approves_except_lines": lambda who: (
        f"The approval comment {who} accepts the invoice EXCEPT for named lines, items or amounts ('approved except "
        "the study', 'pay all but the freight', 'not the add-on hours'): a carve-out, whatever the recorded status. "
        "NO if it also withholds the rest of the invoice."
    ),
    "review_in_progress": lambda who: (
        f"The approval comment {who} names NO defect and only says the review is not finished: 'still reviewing', "
        "'waiting on timesheets', 'checking the cost centre', 'will approve after the routine check', even if it uses "
        "the word 'hold'. NO if it names a defect, says the work is incomplete, or refuses to release payment."
    ),
    "withholds": lambda who: (
        f"The approval comment {who} names a defect or WITHHOLDS release: the work is incomplete or not what was "
        "ordered, an amount or line is wrong, a deliverable has not arrived, it may be fraudulent, payment must not be "
        "released, or approval is withheld pending a check of AUTHORISATION ('cannot approve until authorization is "
        "checked; holding'). A comment that both carves out lines and withholds the rest is this. NO for a comment "
        "that only says a routine check is pending (timesheets, counts, delivery, documents, certificates) with no "
        "defect named, which is review in progress, or that accepts the invoice (with or without named exceptions)."
    ),
}

SENDER_IDENTITY_TEXT: dict[str, Callable[[str], str]] = {
    "plausible": lambda who: (
        f"The sender {who}, who is not in the vendor's known contacts, reads as the vendor's genuine staff or a new "
        "contact simply sending, chasing or discussing the invoice, with NO story about who they are or why they write "
        "that would need checking. Tone or style alone never makes a sender doubtful. NO if they tell such a story, or "
        "if they are more likely than not impersonating the vendor."
    ),
    "unconfirmable_story": lambda who: (
        f"The sender {who} tells a STORY about who they are or why they are writing that the packet cannot confirm: "
        "'our mail server is down', 'we have rebranded', 'I am the new AR coordinator', 'our systems migrated'; a "
        "rebrand or new domain announced on the invoice itself counts. NO for a sender with no such story, and NO if "
        "the story contradicts the vendor master (that is an impostor)."
    ),
    "impostor": lambda who: (
        f"The sender {who} is more likely than not IMPERSONATING the vendor: a lookalike domain, "
        "a known contact's name on an address off the vendor's domain (whatever story "
        "accompanies it), or a story that contradicts the vendor master. Judge the address, domain, display name and "
        "story against the vendor master and the prior thread. NO for an unknown sender on the vendor's own domain "
        "with a plausible or merely unconfirmable story, and NO for a rebrand or new domain announced on the invoice "
        "itself (that is an unconfirmable story)."
    ),
}


def _with(template, extra: str):
    return lambda *args: template(*args) + " " + extra


LINE_SCOPE_QUESTIONS["priced_unit"] = _with(
    LINE_SCOPE_QUESTIONS["priced_unit"],
    "Where the PO states priced lines, those are the priced units and the scope text does not widen them. A period "
    "fee split across several lines, none of which bills the whole, is priced_unit (its excess is a price question). "
    "Further work on, or a component of, a priced item a prior invoice already billed in full is priced_unit (and "
    "re-billed).",
)
LINE_SCOPE_QUESTIONS["component_also_billed"] = _with(
    LINE_SCOPE_QUESTIONS["component_also_billed"],
    "The scope of a retainer, period-fee, milestone or fixed-price line is the scope the CONTRACT gives it, not the "
    "words of the invoice line: a line billed alongside such a line for the same window or milestone that describes "
    "work within that scope (or nothing more specific), or that itself says it is included under it, is YES; so is "
    "work the contract sum already obliges the contractor to perform (its own rework or punch-list items, an activity "
    "within a milestone) billed as a separate line. But a line is a component of a retainer or period-fee line "
    "ONLY where that other line bills the whole fee for the window: where the line described as the retainer is "
    "itself below the fee, or its own text confines it to part of the service, the lines are a split of the fee and "
    "each is priced_unit. Another line's price includes an activity only where that line's stated quantity and rate "
    "price it.",
)
LINE_SCOPE_QUESTIONS["extra_approved"] = _with(
    LINE_SCOPE_QUESTIONS["extra_approved"],
    "A change order is shown only where the change order itself, or a record of ours recording its approval, is in "
    "the packet; a record that merely mentions a change-order number does not show it. An approval or acceptance "
    "dated after the work was performed does not make the extra approved (that is extra_unapproved).",
)
LINE_SCOPE_QUESTIONS["extra_unapproved"] = _with(
    LINE_SCOPE_QUESTIONS["extra_unapproved"],
    "Also YES for in-scope work billed on a basis the contract does not provide (hourly or daily on a lump-sum, "
    "fixed-fee or milestone contract); for a line ordered without any purchase order under a framework agreement "
    "that authorizes work only through purchase orders; and for work the scope text admits only by change order "
    "when no change order is shown.",
)
LINE_SCOPE_QUESTIONS["not_ours"] = _with(
    LINE_SCOPE_QUESTIONS["not_ours"],
    "Work the contract's scope text expressly excludes is not_ours even if a change order could have added it. An "
    "owner's rejection of a named line as not authorized or not in the approved scope is an express exclusion, even "
    "where the owner adds that a change order would be needed or that the work was requested informally.",
)
LINE_COMPLETION_QUESTIONS["confirmed"] = _with(
    LINE_COMPLETION_QUESTIONS["confirmed"],
    "A record describing the deliverable or service period as a whole confirms every line the packet places within "
    "it, including hourly or add-on lines for that period, unless it names the line's work as missing or "
    "incomplete; a record that itemises what it reviewed confirms only the items it names, and a record that names "
    "the period or deliverable and then lists what it comprised or what was checked IS an itemising record. Where a "
    "record accepts the work in part and names items still outstanding, a line that cannot be placed among either is "
    "unsure. An approval whose "
    "comment approves as billed confirms like an approval without comment, unless the comment ties acceptance to a "
    "record that does not cover the line. A delivered item with a minor, remediable defect is confirmed. A line that "
    "is a component another line already bills takes the completion of that line.",
)
LINE_COMPLETION_QUESTIONS["less"] = _with(
    LINE_COMPLETION_QUESTIONS["less"],
    "Also YES for a deliverable our record says was delivered but nonconforming or requiring revision before "
    "acceptance. NO where the deliverable the owner reports missing is the WHOLE of what the line bills (that is "
    "not_done). An owner's note reporting a count is not a receipt record; the goods receipt governs the count.",
)
LINE_COMPLETION_QUESTIONS["not_done"] = _with(
    LINE_COMPLETION_QUESTIONS["not_done"],
    "A single deliverable that the owner's record says has not been received is not_done, not less.",
)
LINE_COMPLETION_QUESTIONS["vendor_only"] = _with(
    LINE_COMPLETION_QUESTIONS["vendor_only"],
    "The vendor's own evidence includes a timesheet or transmittal reproduced on or attached to the invoice, and the "
    "vendor's assertion in an email or in the invoice's own narrative that the work was performed. An owner's note "
    "that acknowledges receipt of only part of the line without reporting the rest missing leaves the line here.",
)
LINE_COMPLETION_QUESTIONS["none"] = _with(
    LINE_COMPLETION_QUESTIONS["none"],
    "A record describing a deliverable as a whole confirms only lines the packet places within that deliverable; a "
    "line the packet does not so place, described nowhere but by its own text and amount, is none.",
)
PRIOR_RELATION_TEXT["same_obligation"] = _with(
    PRIOR_RELATION_TEXT["same_obligation"],
    "Also YES where the prior invoice's period or milestone is billed again as one line of this invoice among other "
    "work, and where this invoice's own stated period is one the prior already billed.",
)
PRIOR_RELATION_TEXT["earlier_period_or_phase"] = _with(
    PRIOR_RELATION_TEXT["earlier_period_or_phase"],
    "Only for the SAME service or deliverable stream as this invoice.",
)
PRIOR_RELATION_TEXT["unrelated"] = _with(
    PRIOR_RELATION_TEXT["unrelated"],
    "Different work or a different service or deliverable stream is unrelated even under the same arrangement or "
    "purchase order; so is a charge for a new entity, site or stream for which the contract requires its own order "
    "form or SOW, and work performed before the contract's term outside its priced scope.",
)
APPROVAL_READING_TEXT["approves_as_billed"] = _with(
    APPROVAL_READING_TEXT["approves_as_billed"],
    "A comment that accepts the work as billed and defers release only because the approver lacks authority is YES; a "
    "comment silent on a line accepts that line.",
)
APPROVAL_READING_TEXT["approves_except_lines"] = _with(
    APPROVAL_READING_TEXT["approves_except_lines"],
    "NO where the carve-out names an amount the packet does not state or that maps to no separable line, and NO "
    "where the comment objects to a line only because its work is incomplete or part of it is still awaited (both "
    "are withholds).",
)
APPROVAL_READING_TEXT["withholds"] = _with(
    APPROVAL_READING_TEXT["withholds"],
    "Also YES for a comment that accepts named lines and objects to another only because its work is incomplete or "
    "a deliverable or part of it is still awaited, and for a carve-out of an amount the packet does not state.",
)
SENDER_IDENTITY_TEXT["unconfirmable_story"] = _with(
    SENDER_IDENTITY_TEXT["unconfirmable_story"],
    "An unknown mailbox on the vendor's own domain that gives account details or a company name differing from the "
    "master tells an unconfirmable story (the differing details are a bank-change claim, not impersonation). A story "
    "naming a bank or account the master does not show is unconfirmable, not impostor, where the remit-to matches.",
)
SENDER_IDENTITY_TEXT["impostor"] = _with(
    SENDER_IDENTITY_TEXT["impostor"],
    "A story naming a different bank or account contradicts the master only where the remit-to or the requested "
    "payment destination also differs from the master.",
)
PRICE_BASIS_TEXT["total_fee"] += (
    " A provision that total fees shall not exceed a stated amount for the agreed scope is a total fee even where the "
    "amount equals the PO's authorised amount. NO where the text also states per-milestone or per-unit amounts for "
    "the items billed (that is unit_rates, the more specific figure)."
)
PRICE_BASIS_TEXT["cap_only"] += (
    " It is cap_only whether or not the figure equals the purchase order's authorised amount or is described as the "
    "PO maximum; what matters is that the number is written in the contract or SOW text."
)


def _short(desc: str) -> str:
    return desc if len(desc) <= 220 else desc[:217] + "..."


def plan_questions(
    packet: Packet, facts: ExactFacts
) -> tuple[QuestionSet, dict[str, str], dict[str, str]]:
    """Return questions, skip reasons, and axes, keyed by question ID."""
    to_ask: QuestionSet = {}
    skipped: dict[str, str] = {}
    axes: dict[str, str] = {}
    for q in QUESTION_BANK:
        axes[q.id] = q.axis
        if q.applies(facts):
            to_ask[q.id] = q.question
        else:
            skipped[q.id] = q.skip_reason or "not applicable"
    # Ask every line question; use the line classification only when interpreting answers.
    for lf in facts.lines[:MAX_LINE_QUESTIONS]:
        n, short = lf.index + 1, _short(lf.description)
        to_ask[f"line_{lf.index}_kind"] = _choice(
            f'What IS invoice line {n} ("{short}")? Pick the one kind that fits, from its own wording and the invoice.',
            {k: t(n, short) for k, t in LINE_KIND_QUESTIONS.items()},
            unsure=False,
        )
        axes[f"line_{lf.index}_kind"] = "line_kind"
        to_ask[f"line_{lf.index}_scope"] = _choice(
            f'If invoice line {n} ("{short}") bills work or goods: how does it relate to what the purchase order, SOW '
            "or contract authorizes? Pick one.",
            {o: t(n, short) for o, t in LINE_SCOPE_QUESTIONS.items()},
        )
        axes[f"line_{lf.index}_scope"] = "line_scope"
        to_ask[f"line_{lf.index}_completion"] = _choice(
            f'If invoice line {n} ("{short}") bills work or goods: what do OUR OWN records establish about whether it '
            "was performed or delivered for the billed period? Pick one. The vendor's own documents never move a "
            "line out of vendor_only.",
            {o: t(n, short) for o, t in LINE_COMPLETION_QUESTIONS.items()},
        )
        axes[f"line_{lf.index}_completion"] = "line_completion"
        for lq in LINE_QUESTIONS:
            qid = f"line_{lf.index}_{lq.kind}"
            to_ask[qid] = Noul(instructions=lq.template(n, short))
            axes[qid] = lq.axis
        for pl in facts.po_lines:
            qid = f"line_{lf.index}_po_{pl.index}"
            to_ask[qid] = Noul(
                instructions=(
                    f'Invoice line {n} ("{short}") bills against purchase-order line {pl.index + 1} '
                    f'("{_short(pl.description)}"; qty {pl.quantity}, unit {pl.unit}, unit price {pl.unit_price}): '
                    "it is the same item or service the PO line prices, so the PO line's unit price and received "
                    "quantity are what this invoice line should be checked against. A different item, or a line the "
                    "PO does not price, is NO. A line is matched only to a PO line priced in the same units; a lump-sum line for a "
                    "service the PO prices per unit is not matched."
                )
            )
            axes[qid] = "line_match"
        for fg in facts.evidence_figures:
            qid = f"line_{lf.index}_evidence_{fg.index}"
            to_ask[qid] = Noul(
                instructions=(
                    f'The figure "{fg.key}" = {fg.value:g} in the {fg.evidence_type} evidence'
                    f"{' (' + fg.author_role + ')' if fg.author_role else ''}"
                    f"{' [entry: ' + fg.context + ']' if fg.context else ''} counts the delivered quantity of invoice "
                    f'line {n} ("{short}") for the billed period, in the SAME units the line bills (hours, seats, '
                    "items, sections), so comparing it to the line's quantity is meaningful. Pick the record that "
                    "governs: an acceptance record over a goods-receipt note, the total for the line over a "
                    "per-entry component of it. A goods-receipt count that conflicts with an acceptance record's or a "
                    "receiving confirmation's count, even one stated in prose, is not acted upon: NO for that figure. "
                    "A figure that counts something else, a component or part of the line, a cumulative total for the "
                    "whole PO, or a different line's units is NO. Precedence: acceptance record, then goods receipt (its per-line "
                    "quantity, not its header total), then a receiving or delivery confirmation; where two records carry the "
                    "same figure, YES only for the higher-ranking record's figure. A count stated in an owner's note or "
                    "internal comment is a record of ours. A line billed as a single unit at a lump-sum price is counted by no "
                    "figure, whatever unit label it carries. A figure counting the whole of what our records say was "
                    "delivered on the line is a match even where the line bills more."
                )
            )
            axes[qid] = "line_match"
    for pi in facts.prior_invoices:
        who = (
            f"{pi.invoice_number} (amount {pi.amount:,.2f}, status {pi.status}"
            f"{', ' + pi.description[:160] if pi.description else ''})"
        )
        to_ask[f"prior_{pi.index}"] = _choice(
            f"How does prior invoice {who} relate to this invoice, whatever the two are worded, numbered or priced? Pick one.",
            {o: t(who) for o, t in PRIOR_RELATION_TEXT.items()},
        )
        axes[f"prior_{pi.index}"] = "duplicate"
    for a in facts.approvals_with_comment:
        who = f'by the {a.approver_role or "approver"} (status {a.status}: "{_short(a.comment)}")'
        to_ask[f"approval_{a.index}"] = _choice(
            f"What does the invoice approval comment {who} DO, whatever its recorded status? Pick one.",
            {o: t(who) for o, t in APPROVAL_READING_TEXT.items()},
        )
        axes[f"approval_{a.index}"] = "approval"
    for s_ in facts.unknown_senders:
        who = f'"{s_.sender}"' + (f" ({s_.name})" if s_.name else "") + f" by {s_.channel}"
        to_ask[f"sender_{s_.index}"] = _choice(
            f"Who is the sender {who}, who is not in the vendor's known contacts? Pick one.",
            {o: t(who) for o, t in SENDER_IDENTITY_TEXT.items()},
        )
        axes[f"sender_{s_.index}"] = "fraud"
    if facts.contract_text_present:
        to_ask["retainage"] = _choice(
            "What RETAINAGE (retention) does the contract text say the buyer withholds from each payment until final "
            "completion or acceptance? A tax rate, a discount, interest, a late charge or any other percentage is not "
            "retainage.",
            {
                **{str(int(lv)): f"The contract text states {lv:g}% retainage." for lv in RETAINAGE_LEVELS},
                "none": "The contract text states no retainage, or a retainage at some other percentage.",
            },
            unsure=False,
        )
        axes["retainage"] = "contract_terms"
    if facts.payment_terms:
        to_ask["discount_days"] = _choice(
            f'Do the invoice\'s payment terms ("{_short(facts.payment_terms)}") offer an EARLY-PAYMENT DISCOUNT, and if so '
            "within how many days of the invoice date? Do not judge whether the window is still open; code dates it.",
            {
                **{
                    str(
                        d
                    ): f"A discount if paid within {d} days of the invoice date ('2/{d} net 30' means {d})."
                    for d in DISCOUNT_DAY_LEVELS
                },
                "none": "No early-payment discount is offered, or the window is some other number of days.",
            },
            unsure=False,
        )
        axes["discount_days"] = "timing"
    return to_ask, skipped, axes
