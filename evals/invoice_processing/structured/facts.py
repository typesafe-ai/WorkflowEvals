"""Compute facts from structured fields; model findings supply prose-dependent classifications."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from .state import Packet

if TYPE_CHECKING:
    from .findings import Findings

VendorTier = Literal["new", "established", "highly_trusted", "high_risk"]
ApprovalState = Literal["approved", "rejected", "pending"]
LineKind = Literal["work_or_goods", "tax", "charge_on_top", "credit_or_discount", "expense_or_retainage"]

INTERNAL_ROLES = {"project_owner", "receiving", "ap", "procurement"}
# Domestic consumption-tax rates; US sales tax is omitted because it varies by state.
STATUTORY_TAX_RATE = {
    "AU": 10.0,
    "NZ": 15.0,
    "GB": 20.0,
    "UK": 20.0,
    "IE": 23.0,
    "DE": 19.0,
    "FR": 20.0,
    "NL": 21.0,
    "BE": 21.0,
    "ES": 21.0,
    "IT": 22.0,
    "CA": 5.0,
    "SG": 9.0,
    "JP": 10.0,
}
COUNTRY_ALIASES = {
    "united states": "US",
    "united kingdom": "GB",
    "australia": "AU",
    "new zealand": "NZ",
    "netherlands": "NL",
    "germany": "DE",
    "france": "FR",
    "canada": "CA",
    "ireland": "IE",
    "belgium": "BE",
    "spain": "ES",
    "italy": "IT",
    "singapore": "SG",
    "japan": "JP",
}
FIXED_FEE_BILLING = {
    "retainer",
    "managed_service_retainer",
    "fixed_fee",
    "fixed_fee_milestones",
    "milestone",
    "subscription",
    "progress_billing",
}
RECURRING_BILLING_TYPES = {"subscription", "recurring", "retainer", "managed_service_retainer"}
INVOICE_DOCUMENT_TYPES = {"invoice", "tax_invoice", "credit_note"}
ROLE_RANK = {
    "project_owner": 0,
    "requester": 0,
    "department_head": 1,
    "dept_head": 1,
    "ap_manager": 1,
    "manager": 1,
    "director": 1,
    "controller": 2,
    "vp_finance": 2,
    "finance_director": 2,
    "cfo": 3,
    "ceo": 4,
    "board": 4,
}
RETAINAGE_LEVELS = (5.0, 10.0, 15.0)


class PoLineFact(BaseModel):

    model_config = ConfigDict(extra="forbid")

    index: int
    description: str
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    received_quantity: float | None = None
    invoiced_quantity: float | None = None


class EvidenceFigure(BaseModel):

    model_config = ConfigDict(extra="forbid")

    index: int
    key: str = Field(description="Field path, e.g. 'lines[1].received_quantity'.")
    value: float
    evidence_type: str
    author_role: str | None = None
    context: str | None = Field(
        default=None,
        description="Sibling text fields of a nested entry (item, sku, description), for the reader.",
    )


class PriorInvoiceFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    invoice_number: str
    amount: float
    status: str | None = None
    description: str | None = None
    po_number: str | None = None


class SenderFact(BaseModel):

    model_config = ConfigDict(extra="forbid")

    index: int
    sender: str
    name: str | None = None
    channel: str
    on_vendor_domain: bool = False
    off_domain: bool = Field(default=False, description="A mailbox whose domain is not the vendor's.")


class ApprovalFact(BaseModel):

    model_config = ConfigDict(extra="forbid")

    index: int
    status: str
    approver_role: str | None = None
    comment: str


class LineFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    description: str
    amount: float
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = Field(default=None, description="As printed on the line.")
    implied_unit_price: float | None = Field(default=None, description="unit_price, else amount / quantity.")
    # Filled by resolve_facts after model answers.
    kind: LineKind | None = Field(default=None, description="Judged line kind (Part A7); None until judged.")
    is_tax_line: bool = False
    is_substantive: bool = Field(default=True, description="work_or_goods or charge_on_top: gets a category.")
    fee_like: bool = Field(
        default=False, description="charge_on_top: only a duplicate or a fee, never scope."
    )
    po_line_index: int | None = Field(default=None, description="Judged PO line this line bills against.")
    po_unit_price: float | None = None
    po_received_quantity: float | None = None
    po_invoiced_quantity: float | None = None
    unit_price_variance_pct: float | None = None
    price_variance_amount: float = 0.0
    price_reference: str | None = Field(default=None, description="po_line when a PO line was matched.")
    evidence_quantity_key: str | None = Field(
        default=None, description="Judged evidence figure for this line."
    )
    evidence_quantity: float | None = None
    quantity_supported_ratio: float | None = Field(
        default=None, description="min(1, evidence / billed). Below 1 means overbilled units."
    )
    rate_differs: bool = Field(
        default=False,
        description="Judged: the line's own text states a rate other than the one it is billed at.",
    )
    completion: str | None = Field(default=None, description="Judged completion state (Part A7).")
    scope: str | None = Field(default=None, description="Judged scope state (Part A7).")


class ExactFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: dt.date = Field(description="Reference date: invoice.received_date, else invoice_date.")

    invoice_total: float
    currency: str
    line_items_sum: float | None
    lines_sum_to_total: bool | None = Field(description="|Σ lines + tax + adjustments − total| <= 1.00.")
    lines: list[LineFact]
    summary_tax: float = Field(default=0.0, description="fields.tax as printed.")
    summary_adjustments_total: float = Field(default=0.0, description="Σ positive summary adjustments.")
    substantive_lines_total: float = Field(description="Σ line amounts excluding tax lines (resolved).")
    positive_adjustments_total: float = Field(
        description="Positive summary adjustments plus positive credit/discount lines (resolved)."
    )
    positive_discount_lines_total: float = Field(
        default=0.0, description="Credit/discount-kind lines with a positive amount (resolved)."
    )
    tax_line_amount: float = Field(default=0.0, description="Σ amounts of tax-kind lines (resolved).")
    tax_double_counted: bool = Field(default=False, description="A tax-kind line exists and fields.tax > 0.")
    implied_tax_rate_pct: float | None = Field(default=None, description="fields.tax / substantive lines, %.")
    expected_tax_rate_pct: float | None = Field(default=None, description="Statutory rate, same country.")
    tax_rate_unexpected_for_country: bool | None = None
    retainage_pct: float | None = Field(default=None, description="Judged retainage level (Part A5).")
    payment_terms: str | None = None
    invoice_date: dt.date | None = None
    discount_window_open: bool = Field(default=False, description="Judged discount days, dated by code.")
    not_confirmed_total: float = Field(
        default=0.0, description="Σ work lines whose completion is vendor_only, none or unsure (resolved)."
    )
    records_contradict_vendor: bool = Field(
        default=False, description="A 'less' line with no figure to carve by (resolved)."
    )
    currency_mismatch: bool | None = None
    contract_text_present: bool = False
    lines_without_price_reference: int = 0
    acceptance_record_present: bool = False

    po_lines: list[PoLineFact] = Field(default_factory=list)
    evidence_figures: list[EvidenceFigure] = Field(default_factory=list)
    prior_invoices: list[PriorInvoiceFact] = Field(default_factory=list)

    document_is_statement_not_invoice: bool = Field(description="From the structured document_type only.")
    invoice_age_days: int
    days_until_due: int | None
    is_overdue: bool | None
    invoice_predates_po: bool | None
    invoice_predates_contract: bool | None
    invoice_before_earliest_evidence: bool | None

    po_exists: bool
    po_referenced: bool
    po_referenced_matches: bool | None
    vendor_id_matched: bool
    vendor_id_matches_po: bool | None
    remaining_po_balance: float | None
    exceeds_po_balance: bool | None
    overrun_amount: float | None
    po_status: str | None
    po_never_approved: bool
    max_unit_price_variance_pct: float | None = None
    price_variance_amount: float = 0.0
    invoiced_quantity_exceeds_received: bool | None
    received_quantity_ratio: float | None

    has_contract: bool
    contract_billing_type: str | None
    contract_is_recurring: bool
    period_fee_applicable: bool = False
    contract_maximum: float | None
    contract_fee_basis: float | None = Field(default=None, description="The structured maximum (3d).")
    cumulative_billed: float | None
    cumulative_exceeds_contract_fee: bool | None
    contract_fee_excess: float | None
    contract_expired: bool | None

    bill_to_entity_matches_po: bool | None
    bill_to_entity_is_company_entity: bool | None
    bill_to_department_matches_po: bool | None
    contract_reference_matches: bool | None

    invoice_number_already_seen: bool
    exact_amount_seen_before: bool
    similar_amount_seen_before: bool
    reversed_prior_with_same_amount: bool
    prior_invoice_count_in_slice: int
    same_obligation_prior: str | None = Field(default=None, description="Judged prior invoice number.")
    same_obligation_same_amount: bool = Field(default=False, description="…and amount within 1% (2b).")
    same_obligation_other_amount: bool = Field(default=False, description="…but amount outside 1% (2d).")
    reversed_prior: bool = Field(
        default=False, description="The judged same-obligation prior was reversed, credited or disputed."
    )
    duplicate_unsure: bool = Field(
        default=False, description="An unsure relation to a paid or pending prior (B3)."
    )

    bank_details_changed: bool | None
    bank_country_changed: bool | None
    account_holder_differs: bool | None
    bank_verification_event_on_file: bool
    bank_change_recently_verified: bool | None
    recent_verified_bank_change_in_log: bool = False
    bank_change_verified_on_file: bool | None
    days_since_last_bank_change: int | None

    company_domain: str
    external_sender_domains: list[str]
    sender_domains_off_file: list[str]
    senders_not_known_contacts: list[str]
    unknown_senders: list[SenderFact] = Field(
        default_factory=list, description="Unknown external senders of any channel (Part A2)."
    )
    sender_new_mailbox_on_known_domain: bool
    non_email_senders: list[str]
    display_name_matches_known_contact_but_address_does_not: bool
    has_external_communications: bool
    has_internal_communications: bool

    purchase_approved: bool | None
    invoice_approved: bool
    invoice_approval_status: ApprovalState | None
    invoice_approval_roles: list[str]
    invoice_approval_rejected: bool
    invoice_approved_after_receipt: bool
    approver_objection: bool = Field(description="Invoice approval pending/rejected with a comment.")
    approver_objection_text: str | None
    approval_is_qualified: bool = Field(description="Invoice approved with a comment.")
    invoice_approved_without_comment: bool = Field(
        default=False, description="Some invoice approval is 'approved' and carries no comment."
    )
    approval_comment: str | None
    approvals_with_comment: list[ApprovalFact] = Field(
        default_factory=list, description="Invoice approvals carrying a comment (Part A6)."
    )
    highest_approved_role: str | None
    highest_approved_role_rank: int
    requester_is_sole_approver: bool

    has_delivery_evidence: bool
    has_internal_delivery_evidence: bool
    has_vendor_only_evidence: bool

    vendor_age_days: int | None
    paid_invoice_count: int
    disputed_invoice_count: int
    reversed_invoice_count: int
    vendor_status: str | None
    vendor_tier: VendorTier
    amount_vs_typical_ratio: float | None
    exceeds_largest_previous_invoice: bool | None


def _domain(address: str) -> str:
    return address.rsplit("@", 1)[1].lower() if "@" in address else address.lower()


def _country(code: str | None) -> str | None:
    if not code:
        return None
    c = code.strip()
    return COUNTRY_ALIASES.get(c.lower(), c.upper() if len(c) <= 3 else None)


def _has_number(data: dict) -> bool:
    for v in data.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return True
        if isinstance(v, dict) and _has_number(v):
            return True
        if isinstance(v, list) and any(isinstance(x, dict) and _has_number(x) for x in v):
            return True
    return False


def _nested_figures(data: dict, prefix: str = "") -> list[tuple[str, float, str | None]]:
    """Extract nested numeric fields with their surrounding text as context."""
    out: list[tuple[str, float, str | None]] = []
    for k, v in data.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(_nested_figures(v, path + "."))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    texts = [str(x) for x in item.values() if isinstance(x, str)]
                    context = " | ".join(texts)[:200] or None
                    for kk, vv in item.items():
                        if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                            out.append((f"{path}[{i}].{kk}", float(vv), context))
                        elif isinstance(vv, dict):
                            out.extend(_nested_figures(vv, f"{path}[{i}].{kk}."))
    return out


def reference_date(packet: Packet) -> dt.date:
    return packet.invoice.received_date or packet.invoice.fields.invoice_date


def _vendor_tier(packet: Packet, age_days: int | None) -> VendorTier:
    vendor = packet.vendor
    hist = vendor.history_summary
    if vendor.status and vendor.status.lower() in {"on_hold", "blocked", "suspended"}:
        return "high_risk"
    if hist.disputed_invoice_count + hist.reversed_invoice_count >= 3 or (
        hist.reversed_invoice_count >= 2 and hist.paid_invoice_count < 5
    ):
        return "high_risk"
    if age_days is None or age_days < 90 or hist.paid_invoice_count < 3:
        return "new"
    if age_days >= 730 and hist.paid_invoice_count >= 24 and hist.disputed_invoice_count == 0:
        return "highly_trusted"
    return "established"


def compute_exact_facts(packet: Packet, as_of: dt.date | None = None) -> ExactFacts:  # noqa: C901
    inv = packet.invoice
    f = inv.fields
    po = packet.purchase_order
    vendor = packet.vendor
    contract = packet.contract
    today = as_of or reference_date(packet)

    lines: list[LineFact] = []
    for i, li in enumerate(inv.line_items):
        unit_price = li.unit_price
        if unit_price is None and li.quantity:
            unit_price = li.amount / li.quantity
        lines.append(
            LineFact(
                index=i,
                description=li.description,
                amount=li.amount,
                quantity=li.quantity,
                unit=li.unit,
                unit_price=li.unit_price,
                implied_unit_price=unit_price,
            )
        )
    po_lines = [
        PoLineFact(
            index=j,
            description=pl.description,
            quantity=pl.authorized_quantity,
            unit=pl.unit,
            unit_price=pl.unit_price,
            received_quantity=pl.received_quantity,
            invoiced_quantity=pl.invoiced_quantity,
        )
        for j, pl in enumerate(po.line_items if po else [])
    ]
    figures: list[EvidenceFigure] = []
    # Use vendor quantity evidence only when buyer records contain no figure.
    ours = [e for e in packet.delivery_evidence if e.author_role != "vendor" and e.structured_data]
    if not any(_has_number(e.structured_data or {}) for e in ours):
        ours = [e for e in packet.delivery_evidence if e.structured_data]
    for e in ours:
        for k, v in (e.structured_data or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                figures.append(
                    EvidenceFigure(
                        index=len(figures),
                        key=k,
                        value=float(v),
                        evidence_type=e.type,
                        author_role=e.author_role,
                    )
                )
    for e in ours:
        for key, value, context in _nested_figures(e.structured_data or {}):
            figures.append(
                EvidenceFigure(
                    index=len(figures),
                    key=key,
                    value=value,
                    evidence_type=e.type,
                    author_role=e.author_role,
                    context=context,
                )
            )
    priors = [
        PriorInvoiceFact(
            index=n,
            invoice_number=p_.invoice_number,
            amount=p_.amount,
            status=p_.status,
            description=p_.description,
            po_number=p_.po_number,
        )
        for n, p_ in enumerate(packet.prior_invoices)
    ]

    lines_sum = sum(li.amount for li in inv.line_items) if inv.line_items else None
    adjustments_total = sum(a.amount for a in f.adjustments)
    adds_up = (
        None if lines_sum is None else abs(lines_sum + (f.tax or 0) + adjustments_total - f.total) <= 1.00
    )
    summary_adjustments = round(sum(a.amount for a in f.adjustments if a.amount > 0), 2)
    vendor_country, company_country = _country(vendor.country), _country(packet.company.country)
    expected_rate = (
        STATUTORY_TAX_RATE.get(vendor_country)
        if vendor_country and vendor_country == company_country
        else None
    )

    doc_type = (inv.document_type or "").lower()
    is_statement = bool(doc_type and doc_type not in INVOICE_DOCUMENT_TYPES)
    age_days = (today - f.invoice_date).days
    days_until_due = None if f.due_date is None else (f.due_date - today).days
    overdue = None if days_until_due is None else days_until_due < 0
    predates_po = (
        None if po is None or po.fields.issued_date is None else f.invoice_date < po.fields.issued_date
    )
    predates_contract = (
        None
        if contract is None or contract.effective_date is None
        else f.invoice_date < contract.effective_date
    )
    evidence_dates = [e.date for e in packet.delivery_evidence if e.date is not None]
    before_evidence = None if not evidence_dates else f.invoice_date < min(evidence_dates)

    if po is not None:
        pf = po.fields
        remaining = pf.authorized_amount - pf.invoiced_to_date
        exceeds = f.total > remaining + 1e-9
        overrun = max(0.0, f.total - remaining)
        po_ref_match = None if f.purchase_order_number is None else f.purchase_order_number == pf.po_number
        vendor_match = None if f.vendor_id is None or pf.vendor_id is None else f.vendor_id == pf.vendor_id
        entity_match = (
            None if f.bill_to_entity is None or pf.entity is None else f.bill_to_entity == pf.entity
        )
        dept_match = (
            None
            if f.bill_to_department is None or pf.department is None
            else f.bill_to_department.lower() == pf.department.lower()
        )
        po_status = pf.status
        cumulative = pf.invoiced_to_date + f.total
    else:
        remaining = exceeds = overrun = po_ref_match = vendor_match = None
        entity_match = dept_match = po_status = None
        paid_prior = [p_.amount for p_ in packet.prior_invoices if p_.status == "paid"]
        cumulative = sum(paid_prior) + f.total if paid_prior else None
    po_never_approved = any(
        a.type == "purchase_approval" and a.status == "pending" and a.approver_id is None
        for a in packet.approvals
    )
    received_qty = [
        e.structured_data.get("received_quantity")
        for e in packet.delivery_evidence
        if e.type == "goods_receipt" and e.structured_data
    ]
    received_qty = [q for q in received_qty if isinstance(q, (int, float))]
    invoiced_qty = [li.quantity for li in inv.line_items if li.quantity is not None]
    if received_qty and invoiced_qty:
        qty_exceeds = sum(invoiced_qty) > sum(received_qty) + 1e-9
        qty_ratio = min(1.0, sum(received_qty) / sum(invoiced_qty)) if sum(invoiced_qty) else None
    else:
        qty_exceeds = qty_ratio = None

    terms = contract.structured_terms if contract else None
    billing_type = terms.billing_type if terms else None
    recurring = bool(billing_type and billing_type.lower() in RECURRING_BILLING_TYPES)
    fixed_fee_billing = bool(billing_type and billing_type.lower() in FIXED_FEE_BILLING)
    currencies = {c for c in ((po.fields.currency if po else None), (terms.currency if terms else None)) if c}
    currency_mismatch = None if not currencies else any(c.upper() != f.currency.upper() for c in currencies)
    contract_max = terms.maximum_amount if terms else None
    cum_exceeds = None if cumulative is None or contract_max is None else cumulative > contract_max + 0.005
    fee_excess = None if cum_exceeds is None else max(0.0, round(cumulative - contract_max, 2))
    expired = None if not contract or contract.end_date is None else f.invoice_date > contract.end_date
    entity_known = (
        None
        if f.bill_to_entity is None or not packet.company.entities
        else f.bill_to_entity in packet.company.entities
    )
    contract_refs = {
        c
        for c in [contract.contract_id if contract else None, po.fields.contract_reference if po else None]
        if c
    }
    contract_ref_match = (
        None if f.contract_reference is None or not contract_refs else f.contract_reference in contract_refs
    )

    live = [p for p in packet.prior_invoices if p.status in ("paid", "pending")]
    number_seen = any(p.invoice_number == f.invoice_number for p in live)
    amounts = {f.total} | ({lines_sum} if lines_sum is not None and not adds_up else set())
    amount_seen = any(abs(p.amount - a) < 0.005 for a in amounts for p in live)
    similar_seen = any(abs(p.amount - a) <= max(10.0, 0.01 * a) for a in amounts for p in live)
    rebill = any(
        p.status in ("reversed", "credited") and abs(p.amount - f.total) < 0.005
        for p in packet.prior_invoices
    )

    remit = f.remit_to
    profile = vendor.payment_profile
    if (
        remit is None
        or profile is None
        or remit.account_masked is None
        or profile.bank_account_masked is None
    ):
        bank_changed = None
    else:
        bank_changed = remit.account_masked != profile.bank_account_masked
    if remit is None or profile is None or remit.bank_country is None or profile.bank_country is None:
        country_changed = None
    else:
        country_changed = remit.bank_country.lower() != profile.bank_country.lower()
    if remit is None or remit.account_holder is None:
        holder_differs = None
    else:
        on_file = {vendor.legal_name.lower()}
        if profile and profile.account_holder:
            on_file.add(profile.account_holder.lower())
        holder_differs = remit.account_holder.lower() not in on_file
    logged_entries = [
        e
        for e in vendor.bank_change_log
        if remit and remit.account_masked and e.new_account_masked == remit.account_masked
    ]
    verification_event = bool(logged_entries)
    recently_verified = None
    if bank_changed is True:
        recently_verified = any(e.verification_method and (today - e.date).days <= 90 for e in logged_entries)
    change_verified = (
        None
        if remit is None or remit.account_masked is None
        else verification_event or (bank_changed is False and vendor.history_summary.paid_invoice_count > 0)
    )
    recent_verified_any = any(
        e.verification_method and (today - e.date).days <= 90 for e in vendor.bank_change_log
    )
    last_change = max((e.date for e in vendor.bank_change_log), default=None)
    days_since_change = None if last_change is None else (today - last_change).days

    company_domain = _domain(packet.company.ap_inbox)
    known_emails = {c.email.lower() for c in vendor.known_contacts}
    known_names = {c.name.lower() for c in vendor.known_contacts if c.name}
    vendor_domain = vendor.email_domain.lower() if vendor.email_domain else None
    external: list[str] = []
    internal = False
    off_file: set[str] = set()
    unknown_senders: set[str] = set()
    sender_facts: list[SenderFact] = []
    non_email: list[str] = []
    spoof_candidate = new_mailbox = False
    for m in packet.communications:
        sender = m.from_.lower()
        if "@" not in sender:
            non_email.append(sender)
            external.append(sender)
            if sender not in unknown_senders:
                unknown_senders.add(sender)
                sender_facts.append(
                    SenderFact(index=len(sender_facts), sender=sender, name=m.from_name, channel=m.channel)
                )
            continue
        domain = _domain(sender)
        if domain == company_domain:
            internal = True
            continue
        external.append(domain)
        if vendor_domain and domain != vendor_domain:
            off_file.add(domain)
        if sender not in known_emails:
            if sender not in unknown_senders:
                sender_facts.append(
                    SenderFact(
                        index=len(sender_facts),
                        sender=sender,
                        name=m.from_name,
                        channel=m.channel,
                        on_vendor_domain=bool(vendor_domain and domain == vendor_domain),
                        off_domain=bool(vendor_domain and domain != vendor_domain),
                    )
                )
            unknown_senders.add(sender)
            if vendor_domain and domain == vendor_domain:
                new_mailbox = True
            if m.from_name and m.from_name.lower() in known_names and domain != vendor_domain:
                spoof_candidate = True

    purchase = [a for a in packet.approvals if a.type == "purchase_approval"]
    purchase_approved = None if not purchase else any(a.status == "approved" for a in purchase)
    inv_approvals = [a for a in packet.approvals if a.type == "invoice_approval"]
    invoice_approved = any(a.status == "approved" for a in inv_approvals)
    invoice_rejected = any(a.status == "rejected" for a in inv_approvals)
    statuses = {a.status for a in inv_approvals}
    approval_status = next((st for st in ("approved", "rejected", "pending") if st in statuses), None)
    roles = sorted({a.approver_role for a in inv_approvals if a.status == "approved" and a.approver_role})
    objections = [
        a for a in inv_approvals if a.status in ("pending", "rejected") and (a.comment or "").strip()
    ]
    received = inv.received_date or f.invoice_date
    approved_after_receipt = any(
        a.status == "approved" and a.timestamp is not None and a.timestamp.date() >= received
        for a in inv_approvals
    )
    approved_with_comment = [a for a in inv_approvals if a.status == "approved" and (a.comment or "").strip()]
    approval_facts = [
        ApprovalFact(index=n, status=a.status, approver_role=a.approver_role, comment=a.comment.strip())
        for n, a in enumerate(a for a in inv_approvals if (a.comment or "").strip())
    ]
    approved_all = [a for a in packet.approvals if a.status == "approved" and a.approver_role]
    highest_role = max(approved_all, key=lambda a: ROLE_RANK.get(a.approver_role, 0), default=None)
    requester = po.fields.requester_id if po else None
    approver_ids = {a.approver_id for a in packet.approvals if a.status == "approved" and a.approver_id}
    sole_requester = bool(requester) and approver_ids == {requester}

    internal_evidence = any(e.author_role in INTERNAL_ROLES for e in packet.delivery_evidence)
    vendor_only = bool(packet.delivery_evidence) and not internal_evidence
    age = None if vendor.vendor_since is None else (today - vendor.vendor_since).days
    hist = vendor.history_summary
    ratio = None if not hist.typical_invoice_amount else f.total / hist.typical_invoice_amount
    exceeds_largest = (
        None if hist.largest_previous_invoice is None else f.total > hist.largest_previous_invoice
    )

    substantive_total = round(sum(lf.amount for lf in lines), 2)  # Include every line until its kind is judged.
    return ExactFacts(
        as_of=today,
        invoice_total=f.total,
        currency=f.currency,
        line_items_sum=lines_sum,
        lines_sum_to_total=adds_up,
        lines=lines,
        summary_tax=f.tax or 0.0,
        summary_adjustments_total=summary_adjustments,
        substantive_lines_total=substantive_total,
        positive_adjustments_total=summary_adjustments,
        expected_tax_rate_pct=expected_rate,
        currency_mismatch=currency_mismatch,
        contract_text_present=bool(contract and (contract.relevant_excerpts or contract.raw_text)),
        acceptance_record_present=any(
            e.type == "acceptance_record" and bool(e.structured_data) for e in packet.delivery_evidence
        ),
        po_lines=po_lines,
        evidence_figures=figures,
        prior_invoices=priors,
        document_is_statement_not_invoice=is_statement,
        invoice_age_days=age_days,
        days_until_due=days_until_due,
        is_overdue=overdue,
        invoice_predates_po=predates_po,
        invoice_predates_contract=predates_contract,
        invoice_before_earliest_evidence=before_evidence,
        po_exists=po is not None,
        po_referenced=f.purchase_order_number is not None,
        po_referenced_matches=po_ref_match,
        vendor_id_matched=f.vendor_id is not None,
        vendor_id_matches_po=vendor_match,
        remaining_po_balance=remaining,
        exceeds_po_balance=exceeds,
        overrun_amount=overrun,
        po_status=po_status,
        po_never_approved=po_never_approved,
        invoiced_quantity_exceeds_received=qty_exceeds,
        received_quantity_ratio=qty_ratio,
        has_contract=contract is not None,
        contract_billing_type=billing_type,
        contract_is_recurring=recurring,
        period_fee_applicable=fixed_fee_billing or recurring,
        contract_maximum=contract_max,
        contract_fee_basis=contract_max,
        cumulative_billed=None if cumulative is None else round(cumulative, 2),
        cumulative_exceeds_contract_fee=cum_exceeds,
        contract_fee_excess=fee_excess,
        contract_expired=expired,
        bill_to_entity_matches_po=entity_match,
        bill_to_entity_is_company_entity=entity_known,
        bill_to_department_matches_po=dept_match,
        contract_reference_matches=contract_ref_match,
        invoice_number_already_seen=number_seen,
        exact_amount_seen_before=amount_seen,
        similar_amount_seen_before=similar_seen,
        reversed_prior_with_same_amount=rebill,
        prior_invoice_count_in_slice=len(packet.prior_invoices),
        bank_details_changed=bank_changed,
        bank_country_changed=country_changed,
        account_holder_differs=holder_differs,
        bank_verification_event_on_file=verification_event,
        bank_change_recently_verified=recently_verified,
        bank_change_verified_on_file=change_verified,
        recent_verified_bank_change_in_log=recent_verified_any,
        days_since_last_bank_change=days_since_change,
        company_domain=company_domain,
        external_sender_domains=sorted(set(external)),
        sender_domains_off_file=sorted(off_file),
        senders_not_known_contacts=sorted(unknown_senders),
        unknown_senders=sender_facts,
        sender_new_mailbox_on_known_domain=new_mailbox,
        non_email_senders=non_email,
        display_name_matches_known_contact_but_address_does_not=spoof_candidate,
        has_external_communications=bool(external),
        has_internal_communications=internal,
        purchase_approved=purchase_approved,
        invoice_approved=invoice_approved,
        invoice_approval_status=approval_status,
        invoice_approval_roles=roles,
        invoice_approval_rejected=invoice_rejected,
        invoice_approved_after_receipt=approved_after_receipt,
        approver_objection=bool(objections),
        approver_objection_text=objections[0].comment if objections else None,
        approval_is_qualified=bool(approved_with_comment),
        invoice_approved_without_comment=any(
            a.status == "approved" and not (a.comment or "").strip() for a in inv_approvals
        ),
        approval_comment=approved_with_comment[0].comment if approved_with_comment else None,
        approvals_with_comment=approval_facts,
        payment_terms=f.payment_terms,
        invoice_date=f.invoice_date,
        highest_approved_role=highest_role.approver_role if highest_role else None,
        highest_approved_role_rank=ROLE_RANK.get(highest_role.approver_role, 0) if highest_role else -1,
        requester_is_sole_approver=sole_requester,
        has_delivery_evidence=bool(packet.delivery_evidence),
        has_internal_delivery_evidence=internal_evidence,
        has_vendor_only_evidence=vendor_only,
        vendor_age_days=age,
        paid_invoice_count=hist.paid_invoice_count,
        disputed_invoice_count=hist.disputed_invoice_count,
        reversed_invoice_count=hist.reversed_invoice_count,
        vendor_status=vendor.status,
        vendor_tier=_vendor_tier(packet, age),
        amount_vs_typical_ratio=ratio,
        exceeds_largest_previous_invoice=exceeds_largest,
    )


def resolve_facts(facts: ExactFacts, findings: Findings) -> ExactFacts:  # noqa: C901
    """Apply findings and recompute dependent amounts and quantities. Idempotent."""
    by_index = {lf.index: lf for lf in findings.lines}
    po_by_index = {pl.index: pl for pl in facts.po_lines}
    fig_by_index = {fg.index: fg for fg in facts.evidence_figures}
    has_receipts = any(
        fg.evidence_type in ("goods_receipt", "acceptance_record") for fg in facts.evidence_figures
    )
    lines: list[LineFact] = []
    for lf in facts.lines:
        j = by_index.get(lf.index)
        upd: dict = {}
        kind = j.kind if j is not None else None
        upd["kind"] = kind
        upd["is_tax_line"] = kind == "tax"
        upd["is_substantive"] = kind in (None, "work_or_goods", "charge_on_top")
        upd["fee_like"] = kind == "charge_on_top"
        upd["rate_differs"] = bool(j.rate_differs) if j is not None else False
        upd["completion"] = j.completion if j is not None and kind == "work_or_goods" else None
        upd["scope"] = j.scope if j is not None and kind == "work_or_goods" else None
        pl = po_by_index.get(j.po_line) if j is not None and j.po_line is not None else None
        if pl is not None and pl.unit_price and lf.unit_price is not None and kind != "tax":
            variance = (lf.unit_price - pl.unit_price) / pl.unit_price
            upd.update(
                po_line_index=pl.index,
                po_unit_price=pl.unit_price,
                po_received_quantity=pl.received_quantity,
                po_invoiced_quantity=pl.invoiced_quantity,
                price_reference="po_line",
                unit_price_variance_pct=round(variance, 4),
                price_variance_amount=round((lf.unit_price - pl.unit_price) * (lf.quantity or 1), 2)
                if variance > 0
                else 0.0,
            )
        else:
            upd.update(
                po_line_index=pl.index if pl is not None else None,
                po_unit_price=None,
                po_received_quantity=pl.received_quantity if pl is not None else None,
                po_invoiced_quantity=pl.invoiced_quantity if pl is not None else None,
                price_reference=None,
                unit_price_variance_pct=None,
                price_variance_amount=0.0,
            )
        # Prefer matched evidence; otherwise use received-minus-invoiced PO quantities.
        fg = fig_by_index.get(j.evidence_figure) if j is not None and j.evidence_figure is not None else None
        ratio = None
        key = None
        qty = None
        if fg is not None and lf.quantity:
            key, qty = fg.key, fg.value
            ratio = min(1.0, fg.value / lf.quantity)
        elif not has_receipts and lf.quantity and upd.get("po_received_quantity") is not None:
            available = max(0.0, upd["po_received_quantity"] - (upd.get("po_invoiced_quantity") or 0.0))
            if available + 1e-9 < lf.quantity:
                key, qty, ratio = "po_received_minus_invoiced", available, round(available / lf.quantity, 4)
        upd.update(evidence_quantity_key=key, evidence_quantity=qty, quantity_supported_ratio=ratio)
        lines.append(lf.model_copy(update=upd))

    tax_lines = [lf for lf in lines if lf.is_tax_line]
    tax_line_amount = round(sum(lf.amount for lf in tax_lines), 2)
    substantive_total = round(sum(lf.amount for lf in lines if not lf.is_tax_line), 2)
    positive_credit_lines = round(
        sum(lf.amount for lf in lines if lf.kind == "credit_or_discount" and lf.amount > 0), 2
    )
    tax_double = bool(tax_lines) and facts.summary_tax > 0
    implied_rate = (
        round(facts.summary_tax / substantive_total * 100, 2)
        if facts.summary_tax and substantive_total > 0
        else None
    )
    tax_unexpected = (
        None
        if implied_rate is None or facts.expected_tax_rate_pct is None
        else abs(implied_rate - facts.expected_tax_rate_pct) > 0.5
    )
    variances = [lf.unit_price_variance_pct for lf in lines if lf.unit_price_variance_pct is not None]

    same_prior = None
    same_amount = other_amount = reversed_prior = dup_unsure = False
    band = max(10.0, 0.01 * facts.invoice_total)
    relations = {pf.index: pf.relation for pf in findings.priors}
    for prior in facts.prior_invoices:
        rel = relations.get(prior.index)
        live = prior.status in ("paid", "pending")
        if rel == "same_obligation" and live:
            if same_prior is None or abs(prior.amount - facts.invoice_total) <= band:
                same_prior = prior.invoice_number
                same_amount = same_amount or abs(prior.amount - facts.invoice_total) <= band
        elif rel == "same_obligation" and not live:
            reversed_prior = True
        elif rel == "unsure" and live:
            dup_unsure = True
    other_amount = same_prior is not None and not same_amount

    discount_open = False
    if findings.discount_days and facts.invoice_date is not None:
        discount_open = facts.as_of <= facts.invoice_date + dt.timedelta(days=int(findings.discount_days))

    not_confirmed = round(
        sum(lf.amount for lf in lines if lf.completion in ("vendor_only", "none", "unsure")), 2
    )
    contradiction = any(
        lf.completion == "less"
        and (lf.quantity_supported_ratio is None or lf.quantity_supported_ratio >= 0.98)
        for lf in lines
    )

    return facts.model_copy(
        update={
            "lines": lines,
            "tax_line_amount": tax_line_amount,
            "tax_double_counted": tax_double,
            "substantive_lines_total": substantive_total,
            "positive_discount_lines_total": positive_credit_lines,
            "positive_adjustments_total": round(facts.summary_adjustments_total + positive_credit_lines, 2),
            "implied_tax_rate_pct": implied_rate,
            "tax_rate_unexpected_for_country": tax_unexpected,
            "retainage_pct": findings.retainage_pct,
            "max_unit_price_variance_pct": max(variances) if variances else None,
            "price_variance_amount": round(sum(lf.price_variance_amount for lf in lines), 2),
            "lines_without_price_reference": sum(
                1 for lf in lines if lf.is_substantive and lf.price_reference is None
            ),
            "same_obligation_prior": same_prior,
            "same_obligation_same_amount": same_amount,
            "same_obligation_other_amount": other_amount,
            "reversed_prior": reversed_prior,
            "duplicate_unsure": dup_unsure,
            "discount_window_open": discount_open,
            "not_confirmed_total": not_confirmed,
            "records_contradict_vendor": contradiction,
        }
    )
