"""Collect holds and corrections before the final release stage; earlier stages cannot release payment."""

from __future__ import annotations

import datetime as dt
from typing import Any

from .actions import Action, apply_precedence, payment_blocked
from .facts import ROLE_RANK, ExactFacts, resolve_facts
from .findings import Findings, LineCategory
from .policy import Decision, PolicyProfile, RuleTrace

SAFETY = {Action.HOLD_VERIFY_BANK_DETAILS, Action.VERIFY_VENDOR_OUT_OF_BAND}
HOLDS_AND_REVIEWS = {
    Action.HOLD_VERIFY_BANK_DETAILS,
    Action.VERIFY_VENDOR_OUT_OF_BAND,
    Action.DUPLICATE_REVIEW,
    Action.PROCUREMENT_REVIEW,
    Action.PROJECT_OWNER_REVIEW,
    Action.HOLD_REQUEST_DOCUMENTS,
    Action.APPROVAL_ESCALATION,
}
VENDOR_FACING = {Action.DISPUTE_LINES, Action.REQUEST_CORRECTED_INVOICE}
# Add-ons require clearer scope evidence; explicit exclusions apply even to vague invoices.


class _Flow:

    def __init__(self, facts: ExactFacts, findings: Findings, profile: PolicyProfile) -> None:
        self.f = facts
        self.fd = findings
        self.p = profile
        self.actions: set[Action] = set()
        self.trace: list[RuleTrace] = []
        self.carved = 0.0
        self.whole_hold: list[str] = []
        self.disputed_lines: list[str] = []
        self.vendor_must_fix: list[str] = []
        self.waiting_on: list[str] = []
        self.role_required: str | None = None
        self.expected_loss: float | None = None

    def node(
        self,
        node_id: str,
        fired: bool,
        adds: list[Action],
        inputs: dict[str, Any] | None = None,
        note: str = "",
    ) -> bool:
        stage = node_id[0]
        self.trace.append(
            RuleTrace(
                rule=node_id,
                step=stage,
                branch=_branch(adds),
                fired=fired,
                actions=adds if fired else [],
                adds=list(adds),
                inputs=inputs or {},
                note=note,
            )
        )
        if fired:
            self.actions |= set(adds)
        return fired

    def carve(self, amount: float, label: str) -> None:
        """Subtract the withheld amount and its share of summary tax."""
        f = self.f
        tax_part = (
            round(amount * (f.implied_tax_rate_pct or 0.0) / 100.0, 2) if f.tax_line_amount == 0 else 0.0
        )
        amount = max(0.0, round(amount + tax_part, 2))
        self.carved = round(self.carved + amount, 2)
        self.disputed_lines.append(label)

    def hold_whole(self, why: str) -> None:
        self.whole_hold.append(why)


def _branch(adds: list[Action]):
    from .actions import BRANCH_OF, Branch

    return BRANCH_OF[adds[0]] if adds else Branch.PAY


def decide(facts: ExactFacts, findings: Findings, profile: PolicyProfile) -> Decision:  # noqa: C901
    facts = resolve_facts(facts, findings)
    fl = _Flow(facts, findings, profile)
    f, fd, p = facts, findings, profile
    total = f.invoice_total
    lines = {lf.index: lf for lf in fd.lines}
    substantive = [lf for lf in f.lines if lf.is_substantive]

    bank_signal = bool(
        (f.bank_details_changed and not f.bank_change_recently_verified)
        or f.account_holder_differs
        or f.bank_country_changed
        or (fd.bank_change_claimed and not f.recent_verified_bank_change_in_log)
    )
    new_mailbox_with_bank = bool(
        f.sender_new_mailbox_on_known_domain and (f.bank_details_changed or fd.bank_change_claimed)
    )
    impostor_signal = bool(
        fd.sender_genuine == "impostor"
        or f.sender_domains_off_file
        or f.display_name_matches_known_contact_but_address_does_not
        or new_mailbox_with_bank
    )
    exact_duplicate = bool(
        (f.invoice_number_already_seen or f.same_obligation_same_amount)
        and not (f.reversed_prior or f.reversed_prior_with_same_amount)
    )
    # Structured entity mismatches reject; judged scope mismatches trigger procurement review.
    wrong_entity = bool(
        f.bill_to_entity_matches_po is False
        or f.vendor_id_matches_po is False
        or f.bill_to_entity_is_company_entity is False
    )
    rejected_with_comment = any(a.status == "rejected" for a in f.approvals_with_comment)
    objection = bool(fd.approver_objects or (f.invoice_approval_rejected and not rejected_with_comment))
    owner_unqualified = f.invoice_approved_without_comment
    readings = {a.index: a.reading for a in fd.approvals}
    approved_effective = owner_unqualified or any(
        readings.get(a.index) in ("approves_as_billed", "approves_except_lines")
        for a in f.approvals_with_comment
        if a.status == "approved"
    )

    # Preserve safety holds even when an early stop fires.
    fl.node(
        "B1",
        bank_signal,
        [Action.HOLD_VERIFY_BANK_DETAILS],
        {
            "bank_details_changed": f.bank_details_changed,
            "bank_change_recently_verified": f.bank_change_recently_verified,
            "holder_or_country_differs": bool(f.account_holder_differs or f.bank_country_changed),
            "bank_change_claimed": fd.bank_change_claimed,
        },
        note="HOLD BANK includes the callback",
    )
    callback_reasons = []
    if f.paid_invoice_count == 0 and total > p.first_payment_verify_amount:
        callback_reasons.append("first payment")
    if f.unknown_senders and total >= p.new_mailbox_callback_amount:
        callback_reasons.append("unknown sender (new mailbox, number or handle)")
    if fd.sender_genuine != "yes":
        callback_reasons.append(f"sender {fd.sender_genuine}")
    if fd.pressure_pattern:
        callback_reasons.append("pressure pattern")
    if f.sender_domains_off_file or f.display_name_matches_known_contact_but_address_does_not:
        callback_reasons.append("sender off the vendor's domain")
    fl.node(
        "B2",
        bool(callback_reasons) and not bank_signal,
        [Action.VERIFY_VENDOR_OUT_OF_BAND],
        {"reasons": callback_reasons, "total": total},
    )

    # Early stops
    fraud = (impostor_signal and bank_signal) or (fd.pressure_pattern and (bank_signal or impostor_signal))
    if fl.node(
        "A1",
        fraud,
        [Action.FRAUD_REVIEW],
        {
            "impostor_signal": impostor_signal,
            "bank_signal": bank_signal,
            "pressure_pattern": fd.pressure_pattern,
        },
    ):
        keep = {Action.FRAUD_REVIEW} | ({Action.REJECT_AS_DUPLICATE} if exact_duplicate else set())
        return _finish(fl, keep, payable=0.0)
    if fl.node(
        "A2",
        exact_duplicate,
        [Action.REJECT_AS_DUPLICATE],
        {
            "invoice_number_already_seen": f.invoice_number_already_seen,
            "same_obligation_prior": f.same_obligation_prior,
            "same_obligation_same_amount": f.same_obligation_same_amount,
        },
    ):
        keep = (fl.actions & SAFETY) | {Action.REJECT_AS_DUPLICATE}
        if wrong_entity:
            keep.add(Action.REJECT_AS_UNAUTHORIZED)
        return _finish(fl, keep, payable=0.0)
    dup_review = bool(f.same_obligation_other_amount or f.duplicate_unsure)
    if fl.node(
        "A3",
        wrong_entity,
        [Action.REJECT_AS_UNAUTHORIZED, Action.REQUEST_CORRECTED_INVOICE],
        {
            "bill_to_entity_matches_po": f.bill_to_entity_matches_po,
            "vendor_id_matches_po": f.vendor_id_matches_po,
            "bill_to_entity_is_company_entity": f.bill_to_entity_is_company_entity,
        },
    ):
        keep = (fl.actions & SAFETY) | {Action.REJECT_AS_UNAUTHORIZED, Action.REQUEST_CORRECTED_INVOICE}
        if dup_review:
            keep.add(Action.DUPLICATE_REVIEW)
        if not f.po_exists:
            keep.add(Action.PROCUREMENT_REVIEW)
        return _finish(fl, keep, payable=0.0)
    if fl.node(
        "A4",
        bool(f.document_is_statement_not_invoice or fd.statement_not_invoice),
        [Action.REQUEST_CORRECTED_INVOICE],
        {
            "document_type_not_invoice": f.document_is_statement_not_invoice,
            "judged_statement": fd.statement_not_invoice,
        },
        note="0c: ask for the invoice itself",
    ):
        keep = (fl.actions & SAFETY) | {Action.REQUEST_CORRECTED_INVOICE}
        fl.vendor_must_fix.append("send the actual invoice, not a statement")
        return _finish(fl, keep, payable=0.0)

    # Holds and reviews
    fl.node(
        "B3",
        dup_review,
        [Action.DUPLICATE_REVIEW],
        {
            "same_obligation_other_amount": f.same_obligation_other_amount,
            "duplicate_unsure": f.duplicate_unsure,
            "legitimate_recurring": fd.legitimate_recurring,
        },
        note="when in doubt on duplicates, hold",
    )
    po_closed = (f.po_status or "").lower() in ("closed", "cancelled", "canceled")
    fl.node(
        "B4.authorization",
        (not f.po_exists) or po_closed,
        [Action.PROCUREMENT_REVIEW],
        {"po_exists": f.po_exists, "po_status": f.po_status},
        note="authorization is never a materiality question",
    )
    fl.node(
        "B4.entity",
        fd.different_entity,
        [Action.PROCUREMENT_REVIEW],
        {"different_entity": fd.different_entity},
        note="the invoice reads as a different project or entity than its PO; procurement confirms",
    )
    stale = (f.invoice_age_days or 0) > p.stale_days and not f.invoice_approved_after_receipt
    predates = bool(f.invoice_predates_po or f.invoice_predates_contract)
    fl.node("B5.objection", bool(objection), [Action.PROJECT_OWNER_REVIEW], {"approver_objects": objection})
    fl.node(
        "B5.approval_unsure",
        fd.approval_unsure and not objection,
        [Action.PROJECT_OWNER_REVIEW],
        {"readings": [a.reading for a in fd.approvals]},
        note="an approval comment nobody can read is the owner's to explain",
    )
    fl.node("B5.stale", stale, [Action.PROJECT_OWNER_REVIEW], {"invoice_age_days": f.invoice_age_days})
    if fl.node(
        "B5.predates",
        predates,
        [Action.PROJECT_OWNER_REVIEW, Action.REQUEST_CORRECTED_INVOICE],
        {
            "invoice_predates_po": f.invoice_predates_po,
            "invoice_predates_contract": f.invoice_predates_contract,
        },
    ):
        fl.hold_whole("invoice predates its authorization")
        fl.vendor_must_fix.append("invoice dated before the authorization existed")
    fl.node(
        "B5.contradiction",
        f.records_contradict_vendor and not objection,
        [Action.PROJECT_OWNER_REVIEW],
        {
            "less_lines": [lf.index + 1 for lf in f.lines if lf.completion == "less"],
            "records_contradict_vendor": f.records_contradict_vendor,
        },
        note="5d: a 'less' line code cannot carve by a figure; one with a figure is Stage C's outcome",
    )
    materiality = p.materiality_gate
    de_minimis = 2 * p.review_cost_usd

    vague_hold = fd.too_vague and total > p.vague_ceiling and not (materiality and total < de_minimis)
    fl.node(
        "B6.vague",
        vague_hold,
        [Action.HOLD_REQUEST_DOCUMENTS],
        {"too_vague": fd.too_vague, "total": total, "vague_ceiling": p.vague_ceiling},
    )
    stake = f.not_confirmed_total
    completion_hold = (
        stake > p.vendor_only_evidence_ceiling
        and not approved_effective
        and not (materiality and stake < de_minimis)
    )
    fl.node(
        "B6.completion",
        completion_hold,
        [Action.HOLD_REQUEST_DOCUMENTS],
        {
            "not_confirmed_lines": [
                lf.index + 1 for lf in f.lines if lf.completion in ("vendor_only", "none", "unsure")
            ],
            "stake": stake,
            "approved": approved_effective,
        },
        note="5c: nothing from our side confirms these lines",
    )

    # Line outcomes
    relax = owner_unqualified and p.owner_approval_relaxes_scope
    out_lines: list[int] = []
    for lf in substantive:
        cat = lines[lf.index].category if lf.index in lines else LineCategory.OK
        lfd = lines.get(lf.index)
        if cat is LineCategory.DUPLICATE_CHARGE:
            fl.node(
                f"C.{lf.index + 1}",
                True,
                [Action.DISPUTE_LINES, Action.REQUEST_CORRECTED_INVOICE],
                {"category": cat},
            )
            fl.carve(lf.amount, f"line {lf.index + 1}: duplicate charge ({lf.amount:,.2f})")
        elif cat is LineCategory.NOT_PERFORMED:
            fl.node(
                f"C.{lf.index + 1}",
                True,
                [Action.DISPUTE_LINES, Action.REQUEST_CORRECTED_INVOICE],
                {"category": cat},
            )
            fl.carve(lf.amount, f"line {lf.index + 1}: not performed ({lf.amount:,.2f})")
        elif cat is LineCategory.DECLINED_BY_OWNER:
            fl.node(f"C.{lf.index + 1}", True, [Action.DISPUTE_LINES], {"category": cat})
            fl.carve(lf.amount, f"line {lf.index + 1}: declined by the owner ({lf.amount:,.2f})")
            out_lines.append(lf.index)
        elif cat is LineCategory.OUT_OF_SCOPE:
            plainly = bool(lfd and lfd.plainly_out)
            counts = plainly or not (vague_hold or relax)
            fl.node(
                f"C.{lf.index + 1}",
                counts,
                [],
                {"category": cat, "plainly_out": plainly, "vague_hold": vague_hold, "owner_relaxes": relax},
                note="out of scope; disputed or routed after the allowance test",
            )
            if counts:
                out_lines.append(lf.index)
        elif cat is LineCategory.UNEXPLAINED_FEE:
            fl.node(f"C.{lf.index + 1}", True, [Action.REQUEST_CORRECTED_INVOICE], {"category": cat})
            fl.carve(lf.amount, f"line {lf.index + 1}: unexplained fee ({lf.amount:,.2f})")
        elif lf.quantity_supported_ratio is not None and lf.quantity_supported_ratio < 0.98:
            fl.node(
                f"C.{lf.index + 1}",
                True,
                [Action.DISPUTE_LINES, Action.REQUEST_CORRECTED_INVOICE],
                {
                    "category": "OK",
                    "completion": lf.completion,
                    "overbilled_vs_evidence": True,
                    "quantity_supported_ratio": lf.quantity_supported_ratio,
                },
            )
            fl.carve(
                round(lf.amount * (1 - lf.quantity_supported_ratio), 2),
                f"line {lf.index + 1}: quantity above evidence",
            )
    out_total = round(sum(f.lines[i].amount for i in out_lines), 2)
    declined = [i for i in out_lines if lines[i].category is LineCategory.DECLINED_BY_OWNER]
    scope_material = not (materiality and min(out_total, total) < de_minimis)
    within_allowance = out_total <= p.short_pay_max_amount and out_total <= p.short_pay_max_share * total
    if fl.node(
        "C.scope.within",
        bool(out_lines) and scope_material and within_allowance,
        [Action.DISPUTE_LINES],
        {"out_lines": [i + 1 for i in out_lines], "out_total": out_total},
        note="3g: within the allowance the lines are disputed and the rest may be short paid",
    ):
        for i in out_lines:
            if i not in declined:
                fl.carve(f.lines[i].amount, f"line {i + 1}: out of scope ({f.lines[i].amount:,.2f})")
    if fl.node(
        "C.scope.above",
        bool(out_lines) and scope_material and not within_allowance,
        [Action.PROCUREMENT_REVIEW],
        {"out_lines": [i + 1 for i in out_lines], "out_total": out_total},
        note="3g: above the allowance procurement decides first",
    ):
        for i in out_lines:
            if i not in declined:
                fl.carve(f.lines[i].amount, f"line {i + 1}: out of scope ({f.lines[i].amount:,.2f})")
    if out_lines and not scope_material:
        fl.node("C.scope.immaterial", True, [], {"out_total": out_total, "de_minimis": de_minimis})
    unsure_lines = [lf.index for lf in substantive if lf.scope == "unsure"]
    unsure_total = round(sum(f.lines[i].amount for i in unsure_lines), 2)
    fl.node(
        "B4.scope_unsure",
        bool(unsure_lines) and not relax and not (materiality and unsure_total < de_minimis),
        [Action.PROCUREMENT_REVIEW],
        {"unsure_lines": [i + 1 for i in unsure_lines], "unsure_total": unsure_total, "owner_relaxes": relax},
        note="a line nobody can place against the authorization is procurement's to place",
    )
    # Materiality uses the stake of findings judged more likely than not.
    soft_stake = max(
        out_total if out_lines else 0.0,
        unsure_total,
        total if fd.too_vague else 0.0,
        stake,
    )
    fl.expected_loss = round(0.5 * min(soft_stake, total), 2)

    # Invoice-level corrections
    if fl.node(
        "D.arithmetic",
        f.lines_sum_to_total is False,
        [Action.REQUEST_CORRECTED_INVOICE],
        {"line_items_sum": f.line_items_sum},
    ):
        fl.hold_whole("line amounts do not add up to the total")
        fl.vendor_must_fix.append("line amounts, tax and adjustments must equal the total")
    tax_twice = bool(f.tax_double_counted and (f.tax_line_amount or 0) > 1.0)
    if fl.node(
        "D.tax",
        tax_twice or fd.tax_two_rates,
        [Action.REQUEST_CORRECTED_INVOICE],
        {
            "tax_double_counted": f.tax_double_counted,
            "tax_two_rates": fd.tax_two_rates,
            "tax_line_amount": f.tax_line_amount,
        },
    ):
        if tax_twice:
            fl.carve(f.tax_line_amount, f"duplicate tax line ({f.tax_line_amount:,.2f})")
        fl.vendor_must_fix.append("charge tax once, at one rate")
    adj_tol = max(p.adjustment_tolerance_abs, p.adjustment_tolerance_pct * f.substantive_lines_total)
    adjustments = (
        f.positive_adjustments_total
    )
    adj_fires = adjustments > 0 and (
        adjustments > adj_tol
        or fd.adjustment_duplicates_line
        or fd.adjustment_unexplained
        or f.positive_discount_lines_total > 0
    )
    if fl.node(
        "D.adjustments",
        adj_fires,
        [Action.REQUEST_CORRECTED_INVOICE],
        {
            "adjustments": adjustments,
            "tolerance": adj_tol,
            "duplicates_line": fd.adjustment_duplicates_line,
            "unexplained": fd.adjustment_unexplained,
            "positive_credit_lines": f.positive_discount_lines_total,
        },
    ):
        fl.carve(adjustments, f"unexplained adjustment ({adjustments:,.2f})")
        fl.vendor_must_fix.append("explain or remove the adjustment")
    over_price = [
        lf
        for lf in f.lines
        if (lf.unit_price_variance_pct or 0) > p.price_variance_pct
        and lf.price_variance_amount > p.price_variance_floor
    ]
    rate_lines = [lf for lf in f.lines if lf.rate_differs and lf.is_substantive]
    if fl.node(
        "D.price",
        (bool(over_price) and not fd.variance_explained_by_split) or bool(rate_lines),
        [Action.REQUEST_CORRECTED_INVOICE],
        {
            "lines_above_po_price": [lf.index + 1 for lf in over_price],
            "lines_above_stated_rate": [lf.index + 1 for lf in rate_lines],
            "max_variance_pct": f.max_unit_price_variance_pct,
            "explained_by_split": fd.variance_explained_by_split,
        },
        note="3b: never pay at the reference price without the owner",
    ):
        fl.hold_whole("unit price above the reference")
        fl.disputed_lines.extend(
            f"line {lf.index + 1}: unit price {lf.unit_price_variance_pct:+.1%} vs {lf.price_reference}"
            for lf in over_price
        )
        fl.disputed_lines.extend(
            f"line {lf.index + 1}: billed above its own stated rate" for lf in rate_lines
        )
        fl.vendor_must_fix.append("unit prices above the price reference")
    above = fd.billed_above_basis
    if fl.node(
        "D.price_text",
        above and fd.price_basis == "unit_rates",
        [Action.REQUEST_CORRECTED_INVOICE],
        {"price_basis": fd.price_basis, "billed_above_basis": above},
    ):
        fl.hold_whole("the contract states a lower rate or amount")
        fl.vendor_must_fix.append("bill the rate or amount the contract states")
    if fl.node(
        "D.period_fee",
        above and fd.price_basis == "period_fee",
        [Action.REQUEST_CORRECTED_INVOICE],
        {"price_basis": fd.price_basis, "billed_above_basis": above},
    ):
        fl.hold_whole("billed above the contract's recurring fee")
        fl.vendor_must_fix.append("bill the contract's recurring fee")
    if fl.node(
        "D.currency",
        f.currency_mismatch is True,
        [Action.REQUEST_CORRECTED_INVOICE],
        {"currency": f.currency},
    ):
        fl.hold_whole("invoice currency differs from the PO/contract")
        fl.vendor_must_fix.append("bill in the PO's currency")
    if fl.node(
        "D.received",
        f.invoiced_quantity_exceeds_received is True and not f.acceptance_record_present,
        [Action.REQUEST_CORRECTED_INVOICE],
        {"received_quantity_ratio": f.received_quantity_ratio},
    ):
        fl.carve(
            round(f.substantive_lines_total * (1 - (f.received_quantity_ratio or 0)), 2),
            "quantities not received",
        )
        fl.vendor_must_fix.append("quantities exceed goods received")
    fl.node(
        "D.department",
        f.bill_to_department_matches_po is False,
        [],
        {},
        note="recode to the PO department; not a hold",
    )

    payable = max(0.0, round(total - fl.carved, 2))
    prior_billed = (f.cumulative_billed or total) - total
    fee_excess = (
        round(prior_billed + payable - f.contract_fee_basis, 2) if f.contract_fee_basis is not None else 0.0
    )
    if fl.node(
        "D.fee",
        fee_excess > 0.005,
        [Action.DISPUTE_LINES, Action.REQUEST_CORRECTED_INVOICE],
        {
            "fee_basis": f.contract_fee_basis,
            "prior_billed": prior_billed,
            "payable_before": payable,
            "excess": fee_excess,
        },
        note="3d: the contract, not the PO, says what is owed",
    ):
        fl.carve(fee_excess, f"cumulative billing above the contract fee ({fee_excess:,.2f})")
        payable = max(0.0, round(total - fl.carved, 2))
    # A prose-only fee limit gives no computable excess, so hold the entire invoice for correction.
    if fl.node(
        "D.fee_text",
        above and fd.price_basis == "total_fee" and fee_excess <= 0.005,
        [Action.DISPUTE_LINES, Action.REQUEST_CORRECTED_INVOICE],
        {"price_basis": fd.price_basis, "structured_fee_basis": f.contract_fee_basis},
        note="3d: the stated total fee is exceeded; the excess is not computable from structure",
    ):
        fl.hold_whole("cumulative billing above the contract's stated total fee")
        fl.vendor_must_fix.append("bill within the contract's total fee")
    cap_overrun = (
        round(payable - f.remaining_po_balance, 2)
        if f.remaining_po_balance is not None and fee_excess <= 0.005
        else 0.0
    )
    accepted = bool(approved_effective or f.acceptance_record_present)
    cap_text = above and fd.price_basis == "cap_only" and cap_overrun <= 0.005
    fl.node(
        "B7",
        (cap_overrun > 0.005 or cap_text) and accepted,
        [Action.APPROVAL_ESCALATION],
        {"cap_overrun": cap_overrun, "cap_in_prose_exceeded": cap_text, "accepted": accepted},
    )
    fl.node(
        "B7.unaccepted",
        (cap_overrun > 0.005 or cap_text) and not accepted,
        [Action.PROJECT_OWNER_REVIEW],
        {"cap_overrun": cap_overrun, "cap_in_prose_exceeded": cap_text},
    )

    # Payment release
    if f.retainage_pct:
        payable = round(payable * (1 - f.retainage_pct / 100), 2)
    required = p.required_role(total)
    tier_unmet = f.highest_approved_role_rank < ROLE_RANK.get(required, 0)
    segregation_unmet = total > p.segregation_amount and f.requester_is_sole_approver
    held = bool(fl.actions & HOLDS_AND_REVIEWS) or bool(fl.whole_hold)
    pending_approval = f.invoice_approval_status == "pending" or (
        f.invoice_approved and not approved_effective
    )
    if fl.node(
        "E1",
        held,
        [],
        {"holds": sorted(a.value for a in fl.actions & HOLDS_AND_REVIEWS), "whole_hold": fl.whole_hold},
        note="nothing is released while the invoice is held",
    ):
        for w in fl.whole_hold:
            fl.waiting_on.append(w)
        if pending_approval and not objection:
            fl.waiting_on.append("owner approval (pending, no objection)")
        return _finish(fl, set(fl.actions), payable=0.0)
    if fl.carved > 0:
        owner_on_file = bool(
            approved_effective
            or (f.approval_is_qualified and fd.approval_carves_out)
            or (f.invoice_approval_rejected and not objection and declined)
        )
        disputed = fl.carved
        allowance_ok = disputed <= p.short_pay_max_amount and disputed <= p.short_pay_max_share * total
        can_short_pay = (
            payable > 0 and owner_on_file and allowance_ok and not tier_unmet and not segregation_unmet
        )
        fl.node(
            "E2",
            can_short_pay,
            [Action.SHORT_PAY],
            {
                "payable": payable,
                "disputed": disputed,
                "owner_on_file": owner_on_file,
                "allowance_ok": allowance_ok,
                "tier_unmet": tier_unmet,
                "segregation_unmet": segregation_unmet,
            },
            note="7a: short pay the undisputed part: owner approved, within the allowance, tier met",
        )
        if can_short_pay:
            return _finish(fl, set(fl.actions), payable=payable)
        if not allowance_ok:
            fl.node(
                "E2.above_allowance",
                True,
                [Action.REQUEST_CORRECTED_INVOICE],
                {"disputed": disputed},
                note="7a: hold the whole invoice for a corrected one",
            )
        if not owner_on_file:
            fl.waiting_on.append("owner approval before the accepted part is paid")
        if tier_unmet or segregation_unmet:
            fl.waiting_on.append("the corrected invoice is routed for signature when it arrives")
        return _finish(fl, set(fl.actions), payable=0.0)
    if fl.node(
        "E3.route",
        tier_unmet or segregation_unmet,
        [Action.HUMAN_APPROVAL],
        {
            "required_role": required,
            "highest_approved_role": f.highest_approved_role,
            "segregation_unmet": segregation_unmet,
        },
    ):
        fl.role_required = required if tier_unmet else "second approver"
        fl.waiting_on.append(f"approval by {fl.role_required}")
        return _finish(fl, {Action.HUMAN_APPROVAL}, payable=0.0)
    pending = pending_approval and not objection
    if fl.node(
        "E3.pending",
        pending,
        [Action.SCHEDULE_PAYMENT],
        {
            "approval_status": f.invoice_approval_status,
            "approved_effective": approved_effective,
            "retainage_pct": f.retainage_pct,
        },
        note="6b: never PAY without an approval",
    ):
        fl.waiting_on.append("owner approval (pending, no objection)")
        return _finish(fl, {Action.SCHEDULE_PAYMENT}, payable=payable, deferred=_due(f))
    discount_open = f.discount_window_open
    due_now = bool(
        f.is_overdue or (f.days_until_due is not None and f.days_until_due <= p.schedule_window_days)
    )
    if fl.node(
        "E3.pay",
        due_now or discount_open,
        [Action.PAY],
        {
            "days_until_due": f.days_until_due,
            "is_overdue": f.is_overdue,
            "discount_open": discount_open,
            "retainage_pct": f.retainage_pct,
        },
    ):
        return _finish(fl, {Action.PAY}, payable=payable)
    fl.node(
        "E3.schedule",
        True,
        [Action.SCHEDULE_PAYMENT],
        {"days_until_due": f.days_until_due, "retainage_pct": f.retainage_pct},
    )
    return _finish(fl, {Action.SCHEDULE_PAYMENT}, payable=payable, deferred=_due(f))


def decide_from_signals(facts: ExactFacts, signals: Any, profile: PolicyProfile) -> Decision:
    from .findings import from_signals

    return decide(facts, from_signals(facts, signals), profile)


def _due(f: ExactFacts) -> dt.date | None:
    if f.days_until_due is None:
        return None
    return f.as_of + dt.timedelta(days=f.days_until_due)


def _finish(fl: _Flow, actions: set[Action], payable: float, deferred: dt.date | None = None) -> Decision:
    ordered = apply_precedence(actions) if actions else [Action.PAY]
    blocked = payment_blocked(ordered)
    return Decision(
        policy=fl.p.name,
        actions=ordered,
        primary_action=ordered[0],
        payment_blocked=blocked,
        payable_now=0.0 if blocked else payable,
        disputed_amount=fl.carved,
        disputed_lines=fl.disputed_lines,
        vendor_must_fix=fl.vendor_must_fix,
        waiting_on=fl.waiting_on,
        approval_role_required=fl.role_required,
        deferred_until=deferred,
        expected_loss_usd=fl.expected_loss,
        trace=fl.trace,
    )
