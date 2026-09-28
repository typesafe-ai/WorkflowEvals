"""Each customer turn is an independent case with a supplied initial assistant state."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Turn:
    speaker: Literal["customer", "assistant"]
    text: str


@dataclass
class Customer:
    tenure_months: int
    vip: bool = False
    prior_contacts_30d: int = 0
    prior_refunds_90d: int = 0


@dataclass
class TransactionOnFile:
    transaction_id: str
    merchant: str
    amount_usd: float
    date: str
    channel: Literal["card_present", "online", "recurring", "atm"]


@dataclass
class RefundOnFile:
    amount_usd: float
    eligible: bool
    already_issued: bool = False


@dataclass
class OrderOnFile:
    order_id: str
    item: str
    status: Literal["processing", "shipped", "delivered", "returned"]
    expected_by: Optional[str] = None


@dataclass
class Account:
    card_id: str = "card-0000"
    card_status: Literal["active", "frozen", "closed"] = "active"
    subscription_active: bool = False
    subscription_monthly_usd: float = 0.0
    transaction: Optional[TransactionOnFile] = None
    refund: Optional[RefundOnFile] = None
    order: Optional[OrderOnFile] = None
    outstanding_fee_usd: float = 0.0


@dataclass
class State:
    intent: Optional[str] = None
    pending: Optional[str] = None
    priority: str = "normal"
    resolved: bool = False
    handed_off: bool = False


@dataclass
class Case:
    case_id: str
    dialog_id: str
    turn: int                              # 0-based customer turn index
    theme: str
    transcript: List[Turn]
    customer: Customer
    account: Account
    state: State = field(default_factory=State)
    note: str = ""                         # Never sent to a model.

    def triage_state(self) -> Dict[str, Any]:
        """Include account summaries only; later nodes receive the relevant record details."""
        a = self.account
        return {
            "conversation": [asdict(t) for t in self.transcript],
            "customer": asdict(self.customer),
            "account_summary": {
                "card_status": a.card_status,
                "transaction_on_file": a.transaction is not None,
                "refund_claim_on_file": a.refund is not None,
                "fee_on_file": a.outstanding_fee_usd > 0,
                "order_on_file": a.order is not None,
                "subscription_active": a.subscription_active,
            },
            "assistant_pending_proposal": self.state.pending,
        }

    def consent_state(self) -> Dict[str, Any]:
        """Include only the pending proposal, its terms, the proposal message, and the reply."""
        proposal_msg = next((t.text for t in reversed(self.transcript[:-1]) if t.speaker == "assistant"), None)
        terms = None
        if self.state.pending == "issue_refund" and self.account.refund:
            terms = {"amount_usd": self.account.refund.amount_usd}
        elif self.state.pending == "cancel_subscription":
            terms = {"monthly_usd": self.account.subscription_monthly_usd}
        elif self.state.pending in ("freeze_card", "open_dispute") and self.account.transaction:
            terms = {"transaction": asdict(self.account.transaction)}
        return {"pending_proposal": self.state.pending, "proposed_terms_on_record": terms,
                "assistant_proposal_message": proposal_msg, "customer_reply": self.latest}

    def fraud_state(self) -> Dict[str, Any]:
        return {"conversation": [asdict(t) for t in self.transcript],
                "ledger_entry": asdict(self.account.transaction) if self.account.transaction else "no matching entry on file",
                "card_status": self.account.card_status}

    def money_state(self) -> Dict[str, Any]:
        claim = None
        if self.account.refund:
            claim = {"kind": "refund_claim", "amount_usd": self.account.refund.amount_usd, "eligible_under_policy": self.account.refund.eligible,
                     "already_refunded": self.account.refund.already_issued}
        elif self.account.outstanding_fee_usd > 0:
            claim = {"kind": "fee", "amount_usd": self.account.outstanding_fee_usd}
        return {"conversation": [asdict(t) for t in self.transcript], "on_file": claim,
                "order": asdict(self.account.order) if self.account.order else None,
                "customer": {"tenure_months": self.customer.tenure_months, "prior_refunds_90d": self.customer.prior_refunds_90d}}

    def retention_state(self) -> Dict[str, Any]:
        return {"conversation": [asdict(t) for t in self.transcript],
                "subscription": {"active": self.account.subscription_active, "monthly_usd": self.account.subscription_monthly_usd},
                "customer": asdict(self.customer)}

    def customer_turns_state(self) -> Dict[str, Any]:
        return {"customer_messages": [t.text for t in self.transcript if t.speaker == "customer"]}

    def assistant_messages(self) -> List[Dict[str, Any]]:
        out, k = [], 0
        for t in self.transcript:
            if t.speaker == "assistant":
                out.append({"number": k, "text": t.text}); k += 1
        return out

    def new_assistant_message_numbers(self) -> List[int]:
        """Assistant messages since the previous customer message."""
        cust = [i for i, t in enumerate(self.transcript) if t.speaker == "customer"]
        start = cust[-2] if len(cust) >= 2 else -1
        out, k = [], 0
        for i, t in enumerate(self.transcript):
            if t.speaker == "assistant":
                if i > start:
                    out.append(k)
                k += 1
        return out

    def integrity_state(self) -> Dict[str, Any]:
        return {"assistant_messages": self.assistant_messages(),
                "record": {"refund_issued": bool(self.account.refund and self.account.refund.already_issued),
                           "card_status": self.account.card_status}}

    @property
    def latest(self) -> str:
        return self.transcript[-1].text

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps({"triage_state": self.triage_state(), "state": asdict(self.state)}, sort_keys=True).encode()).hexdigest()[:16]

    def to_json(self) -> dict:
        d = asdict(self)
        d["case_hash"] = self.content_hash()
        return d

    @staticmethod
    def from_json(d: dict) -> "Case":
        acct = d["account"]
        account = Account(
            card_id=acct["card_id"], card_status=acct["card_status"], subscription_active=acct["subscription_active"],
            subscription_monthly_usd=acct["subscription_monthly_usd"], outstanding_fee_usd=acct["outstanding_fee_usd"],
            transaction=TransactionOnFile(**acct["transaction"]) if acct.get("transaction") else None,
            refund=RefundOnFile(**acct["refund"]) if acct.get("refund") else None,
            order=OrderOnFile(**acct["order"]) if acct.get("order") else None,
        )
        return Case(
            case_id=d["case_id"], dialog_id=d["dialog_id"], turn=d["turn"], theme=d["theme"],
            transcript=[Turn(**t) for t in d["transcript"]], customer=Customer(**d["customer"]), account=account,
            state=State(**d["state"]), note=d.get("note", ""),
        )


@dataclass
class DialogTurn:
    customer: str
    state_before: State = field(default_factory=State)
    assistant: Optional[str] = None
    note: str = ""


@dataclass
class Dialog:
    dialog_id: str
    theme: str
    customer: Customer
    account: Account
    turns: List[DialogTurn]
    opening: Optional[str] = None

    def cases(self) -> List[Case]:
        out: List[Case] = []
        transcript: List[Turn] = []
        if self.opening:
            transcript.append(Turn("assistant", self.opening))
        for i, t in enumerate(self.turns):
            transcript.append(Turn("customer", t.customer))
            out.append(Case(
                case_id=f"{self.dialog_id}/t{i}", dialog_id=self.dialog_id, turn=i, theme=self.theme,
                transcript=list(transcript), customer=self.customer, account=self.account,
                state=t.state_before, note=t.note,
            ))
            if t.assistant:
                transcript.append(Turn("assistant", t.assistant))
        return out
