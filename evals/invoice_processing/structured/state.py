from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdditionalField(_Strict):
    name: str
    value: Any


class Company(_Strict):
    company_id: str
    name: str
    country: str | None = None
    industry: str | None = None
    entities: list[str] = Field(
        default_factory=list,
        description="Legal entities that can appear as bill-to or PO entity; first is the parent.",
    )
    ap_inbox: str = Field(description="Address invoices arrive at; separates internal from external senders.")


class Adjustment(_Strict):
    label: str
    amount: float


class RemitTo(_Strict):
    bank_name: str | None = None
    bank_country: str | None = None
    account_masked: str | None = None
    account_holder: str | None = None


class InvoiceFields(_Strict):
    invoice_number: str
    invoice_date: dt.date
    due_date: dt.date | None = None
    vendor_name: str
    vendor_id: str | None = Field(default=None, description="null if AP could not match a vendor record")
    currency: str
    subtotal: float | None = None
    tax: float | None = None
    adjustments: list[Adjustment] = Field(default_factory=list)
    total: float = Field(description="Total as printed, even if it does not add up.")
    purchase_order_number: str | None = None
    contract_reference: str | None = None
    payment_terms: str | None = None
    service_period: str | None = Field(default=None, description="Free text as printed.")
    bill_to_entity: str | None = None
    bill_to_department: str | None = None
    remit_to: RemitTo | None = None


class InvoiceLineItem(_Strict):
    description: str
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    amount: float


class Invoice(_Strict):
    fields: InvoiceFields
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    raw_text: str | None = None
    document_type: str | None = Field(
        default=None,
        description="invoice, statement_of_account, credit_note, pro_forma, reminder, ...",
    )
    source_format: str | None = Field(
        default=None, description="pdf_text, ocr_scan, email_body, e_invoice_xml, csv_export, ..."
    )
    received_date: dt.date | None = None
    additional_fields: list[AdditionalField] = Field(default_factory=list)


class PurchaseOrderFields(_Strict):
    po_number: str
    vendor_id: str | None = None
    vendor_name: str | None = None
    issued_date: dt.date | None = None
    currency: str | None = None
    authorized_amount: float
    invoiced_to_date: float = 0
    status: str | None = Field(default=None, description="open, closed, cancelled, ...")
    entity: str | None = None
    department: str | None = None
    cost_center: str | None = None
    requester_id: str | None = None
    contract_reference: str | None = None


class PurchaseOrderLineItem(_Strict):
    description: str
    authorized_quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    authorized_amount: float | None = None
    received_quantity: float | None = Field(default=None, description="null for service lines")
    invoiced_quantity: float | None = None


class PurchaseOrder(_Strict):
    fields: PurchaseOrderFields
    line_items: list[PurchaseOrderLineItem] = Field(default_factory=list)
    scope_text: str | None = None
    raw_text: str | None = None
    additional_fields: list[AdditionalField] = Field(default_factory=list)


class ContractTerms(_Strict):
    billing_type: str | None = Field(
        default=None, description="milestone, time_and_materials, fixed_fee, subscription, ..."
    )
    maximum_amount: float | None = None
    currency: str | None = None
    entity: str | None = None


class Contract(_Strict):
    contract_id: str
    title: str | None = None
    effective_date: dt.date | None = None
    end_date: dt.date | None = None
    structured_terms: ContractTerms | None = None
    relevant_excerpts: list[str] = Field(default_factory=list)
    raw_text: str | None = None


class PaymentProfile(_Strict):
    currency: str | None = None
    payment_method: str | None = None
    bank_name: str | None = None
    bank_country: str | None = None
    bank_account_masked: str | None = None
    account_holder: str | None = None


class BankChangeLogEntry(_Strict):
    date: dt.date
    previous_account_masked: str | None = None
    new_account_masked: str | None = None
    verification_method: str | None = None
    verified_by_role: str | None = None


class HistorySummary(_Strict):
    paid_invoice_count: int = 0
    typical_invoice_amount: float | None = None
    largest_previous_invoice: float | None = None
    last_invoice_date: dt.date | None = None
    disputed_invoice_count: int = 0
    reversed_invoice_count: int = 0


class KnownContact(_Strict):
    name: str | None = None
    email: str
    role: str | None = None
    since: dt.date | None = None


class Vendor(_Strict):
    vendor_id: str
    legal_name: str
    display_name: str | None = None
    vendor_since: dt.date | None = None
    status: str | None = Field(default=None, description="active, inactive, on_hold, ...")
    country: str | None = None
    email_domain: str | None = None
    payment_profile: PaymentProfile | None = None
    bank_change_log: list[BankChangeLogEntry] = Field(default_factory=list)
    history_summary: HistorySummary = Field(default_factory=HistorySummary)
    known_contacts: list[KnownContact] = Field(default_factory=list)
    notes: str | None = None
    additional_fields: list[AdditionalField] = Field(default_factory=list)


PriorInvoiceStatus = Literal["paid", "pending", "reversed", "disputed", "credited"]


class PriorInvoice(_Strict):
    invoice_number: str
    invoice_date: dt.date
    amount: float
    currency: str | None = None
    description: str | None = None
    po_number: str | None = None
    status: PriorInvoiceStatus
    paid_date: dt.date | None = None
    paid_to_account_masked: str | None = None
    raw_text: str | None = None


Channel = Literal["email", "portal_message", "chat", "voicemail_transcript", "letter"]


class Communication(_Strict):
    message_id: str
    timestamp: dt.datetime
    channel: Channel
    from_: str = Field(
        alias="from", description="Address as it appeared; handle or phone for chat/voicemail."
    )
    from_name: str | None = None
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    subject: str | None = None
    body: str
    attachments: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)


EvidenceType = Literal[
    "goods_receipt",
    "project_note",
    "acceptance_record",
    "delivery_confirmation",
    "service_log",
    "internal_comment",
]
AuthorRole = Literal["project_owner", "receiving", "ap", "procurement", "vendor", "third_party"]


class DeliveryEvidence(_Strict):
    type: EvidenceType
    date: dt.date | None = None
    author_role: AuthorRole | None = None
    structured_data: dict[str, Any] | None = None
    text: str | None = None


ApprovalType = Literal["purchase_approval", "invoice_approval"]
ApproverRole = Literal["project_owner", "department_head", "ap_manager", "controller", "cfo"]
ApprovalStatus = Literal["approved", "pending", "rejected"]


class Approval(_Strict):
    type: ApprovalType
    approver_role: ApproverRole | None = None
    approver_id: str | None = None
    approver_name: str | None = None
    timestamp: dt.datetime | None = None
    status: ApprovalStatus
    comment: str | None = None


class Packet(_Strict):

    schema_version: Literal["1.0"]
    case_id: str
    company: Company
    invoice: Invoice
    purchase_order: PurchaseOrder | None
    contract: Contract | None
    vendor: Vendor
    prior_invoices: list[PriorInvoice] = Field(default_factory=list)
    communications: list[Communication] = Field(default_factory=list)
    delivery_evidence: list[DeliveryEvidence] = Field(default_factory=list)
    approvals: list[Approval] = Field(default_factory=list)

    def as_state(self) -> dict[str, Any]:
        """Preserve authored aliases and values in the model input."""
        return self.model_dump(mode="json", by_alias=True)


class ProfileActions(BaseModel):

    model_config = ConfigDict(extra="allow")

    primary: str | None = None
    actions: list[str]


class Sidecar(_Strict):
    """Reference annotations for scoring; never sent to the model."""

    case_id: str
    family: str | None = None
    truth: dict[str, Any] = Field(default_factory=dict)
    intended_bands: dict[str, str] = Field(default_factory=dict)
    band_probabilities: dict[str, float | None] = Field(
        default_factory=dict,
        description=(
            "Generator's probability per question implied by its band; None when n/a. Oracle input."
        ),
    )
    exact_facts: dict[str, Any] = Field(
        default_factory=dict,
        description=("Reference derivation from the packet; compared against ours as a consistency check."),
    )
    actions_by_profile: dict[str, ProfileActions] = Field(default_factory=dict)
    parent_case_id: str | None = None
    mutation: str | dict[str, Any] | None = None
    consistency_flags: list[str] = Field(default_factory=list)
    annotator_notes: str | None = None

    model_config = ConfigDict(extra="allow")


Case = Packet
