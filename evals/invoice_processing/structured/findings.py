from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from .facts import RETAINAGE_LEVELS, ExactFacts, LineKind

DECIDED = 0.5
DISCOUNT_DAY_LEVELS: tuple[int, ...] = (7, 10, 15, 20, 30)

LINE_KINDS: tuple[LineKind, ...] = (
    "work_or_goods",
    "tax",
    "charge_on_top",
    "credit_or_discount",
    "expense_or_retainage",
)
ScopeState = Literal[
    "priced_unit", "component_also_billed", "extra_approved", "extra_unapproved", "not_ours", "unsure"
]
CompletionState = Literal["confirmed", "less", "not_done", "vendor_only", "none", "unsure"]
PriorRelation = Literal["same_obligation", "earlier_period_or_phase", "unrelated", "unsure"]
PriceBasis = Literal["total_fee", "period_fee", "unit_rates", "cap_only", "none", "unsure"]
ApprovalReading = Literal[
    "approves_as_billed", "approves_except_lines", "review_in_progress", "withholds", "unsure"
]
SenderIdentity = Literal["plausible", "unconfirmable_story", "impostor", "unsure"]
SenderGenuine = Literal["yes", "doubtful", "impostor"]

SCOPE_OPTIONS = tuple(o for o in get_args(ScopeState) if o != "unsure")
COMPLETION_OPTIONS = tuple(o for o in get_args(CompletionState) if o != "unsure")
PRIOR_OPTIONS = tuple(o for o in get_args(PriorRelation) if o != "unsure")
PRICE_BASIS_OPTIONS = tuple(o for o in get_args(PriceBasis) if o not in ("unsure", "none"))
APPROVAL_OPTIONS = tuple(o for o in get_args(ApprovalReading) if o != "unsure")
SENDER_OPTIONS = tuple(o for o in get_args(SenderIdentity) if o != "unsure")


class LineCategory(StrEnum):

    DUPLICATE_CHARGE = "DUPLICATE_CHARGE"
    DECLINED_BY_OWNER = "DECLINED_BY_OWNER"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    NOT_PERFORMED = "NOT_PERFORMED"
    UNEXPLAINED_FEE = "UNEXPLAINED_FEE"
    OK = "OK"


class LineFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    kind: LineKind | None = Field(default=None, description="What the line is (Part A7); None = not judged.")
    scope: ScopeState = Field(default="priced_unit", description="Relation to the priced units (A7).")
    completion: CompletionState = Field(default="confirmed", description="What our records say (A7).")
    rebilled: bool = Field(default=False, description="A paid/pending prior already billed this line (A3).")
    owner_declined: bool = Field(default=False, description="Excluded by name by the owner (A7.2).")
    unexplained_fee: bool = Field(default=False, description="An add-on charge tracing to nothing (A7.5).")
    po_line: int | None = Field(default=None, description="Index of the PO line this line bills against.")
    evidence_figure: int | None = Field(
        default=None, description="Index of the evidence figure counting this line."
    )
    rate_differs: bool = Field(
        default=False, description="The line's own text states a rate other than the billed one."
    )
    why: str = ""

    @property
    def category(self) -> LineCategory:
        if self.kind in ("tax", "credit_or_discount", "expense_or_retainage"):
            return LineCategory.OK
        if self.scope == "component_also_billed" or self.rebilled:
            return LineCategory.DUPLICATE_CHARGE
        if self.kind == "charge_on_top":  # Charges on the work are fees or duplicates, not scope findings.
            return LineCategory.UNEXPLAINED_FEE if self.unexplained_fee else LineCategory.OK
        if self.owner_declined:
            return LineCategory.DECLINED_BY_OWNER
        if self.scope in ("extra_unapproved", "not_ours"):
            return LineCategory.OUT_OF_SCOPE
        if self.completion == "not_done":
            return LineCategory.NOT_PERFORMED
        if self.unexplained_fee:
            return LineCategory.UNEXPLAINED_FEE
        return LineCategory.OK

    @property
    def plainly_out(self) -> bool:
        return self.scope == "not_ours"

    @property
    def scope_unsure(self) -> bool:
        return self.kind == "work_or_goods" and self.scope == "unsure"


class PriorFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    relation: PriorRelation = "unrelated"


class ApprovalFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    reading: ApprovalReading = "approves_as_billed"


class SenderFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    identity: SenderIdentity = "plausible"


class Findings(BaseModel):

    model_config = ConfigDict(extra="forbid")

    source: str = Field(default="model", description="'model' (harness answers) or 'label' (findings form).")
    lines: list[LineFinding] = Field(default_factory=list)
    priors: list[PriorFinding] = Field(default_factory=list)
    approvals: list[ApprovalFinding] = Field(default_factory=list)
    senders: list[SenderFinding] = Field(default_factory=list)
    statement_not_invoice: bool = False
    discount_days: int | None = Field(default=None, description="Early-payment discount window, days.")
    pressure_pattern: bool = False
    bank_change_claimed: bool = False
    different_entity: bool = False
    price_basis: PriceBasis = "none"
    billed_above_basis: bool = False
    variance_explained_by_split: bool = False
    retainage_pct: float | None = Field(
        default=None, description="Retainage level the contract withholds, if any."
    )
    too_vague: bool = False
    adjustment_unexplained: bool = False
    adjustment_duplicates_line: bool = Field(
        default=False, description="A summary adjustment charges what a line already charges."
    )
    tax_two_rates: bool = Field(default=False, description="Tax is charged at two different rates.")

    def line(self, index: int) -> LineFinding | None:
        return next((lf for lf in self.lines if lf.index == index), None)

    @property
    def sender_genuine(self) -> SenderGenuine:
        ids = {s.identity for s in self.senders}
        if "impostor" in ids:
            return "impostor"
        if ids & {"unconfirmable_story", "unsure"}:
            return "doubtful"
        return "yes"

    @property
    def approver_objects(self) -> bool:
        return any(a.reading == "withholds" for a in self.approvals)

    @property
    def approval_unsure(self) -> bool:
        return any(a.reading == "unsure" for a in self.approvals)

    @property
    def approval_covers_invoice(self) -> bool | None:
        if not self.approvals:
            return None
        return any(a.reading == "approves_as_billed" for a in self.approvals) and not self.approver_objects

    @property
    def approval_carves_out(self) -> bool:
        return any(a.reading == "approves_except_lines" for a in self.approvals)

    @property
    def approval_in_progress_only(self) -> bool:
        return bool(self.approvals) and all(a.reading == "review_in_progress" for a in self.approvals)

    @property
    def legitimate_recurring(self) -> bool:
        rels = {p.relation for p in self.priors}
        return "earlier_period_or_phase" in rels and "same_obligation" not in rels

    @property
    def work_confirmed(self) -> bool:
        return all(lf.completion == "confirmed" for lf in self.lines if lf.kind == "work_or_goods")


JUDGEMENT_KEYS: tuple[str, ...] = tuple(
    k for k in Findings.model_fields if k not in ("source", "lines", "priors", "approvals", "senders")
)
LINE_JUDGEMENT_KEYS: tuple[str, ...] = (
    "kind",
    "scope",
    "completion",
    "rebilled",
    "owner_declined",
    "unexplained_fee",
    "po_line",
    "evidence_figure",
    "rate_differs",
)
STATE_JUDGEMENTS: dict[str, tuple[str, ...]] = {
    "scope": SCOPE_OPTIONS,
    "completion": COMPLETION_OPTIONS,
    "prior_relation": PRIOR_OPTIONS,
    "price_basis": PRICE_BASIS_OPTIONS,
    "approval_comment": APPROVAL_OPTIONS,
    "sender_identity": SENDER_OPTIONS,
}


def _argmax(probs: dict[str, float | None], ids: list[tuple[str, Any]]) -> Any:
    """Return the highest-probability option only when it exceeds the decision threshold."""
    best_p, best = 0.0, None
    for qid, option in ids:
        p = probs.get(qid) or 0.0
        if p > best_p:
            best_p, best = p, option
    return best if best_p >= DECIDED else None


def _choice(probs: dict[str, float | None], prefix: str, options: tuple[str, ...]) -> str | None:
    scored = [(probs.get(f"{prefix}_{o}"), o) for o in options]
    scored = [(p, o) for p, o in scored if p is not None]
    if not scored:
        return None
    return max(scored, key=lambda x: x[0])[1]


def _state(
    probs: dict[str, float | None],
    prefix: str,
    options: tuple[str, ...],
    asked: bool = True,
    floor: float = 0.0,
) -> str:
    """Read a choice as unsure when its best probability is below the profile floor."""
    if not asked:
        return "unsure"
    chosen = _choice(probs, prefix, (*options, "unsure"))
    if chosen is None or chosen == "unsure":
        return "unsure"
    return chosen if (probs.get(f"{prefix}_{chosen}") or 0.0) >= floor else "unsure"


def from_signals(facts: ExactFacts, signals: Any, profile: Any = None) -> Findings:
    """Apply the profile thresholds to model probabilities; default to 0.5 without a profile."""
    probs: dict[str, float | None] = getattr(signals, "probabilities", {}) or {}
    decided = getattr(profile, "decided", DECIDED)
    clear_decided = getattr(profile, "clear_decided", DECIDED)
    floor = getattr(profile, "state_floor", 0.0)

    def p(qid: str) -> float | None:
        return probs.get(qid)

    def yes(qid: str) -> bool:
        return (p(qid) or 0.0) >= decided

    def clear(qid: str) -> bool:
        return (p(qid) or 0.0) >= clear_decided

    lines: list[LineFinding] = []
    for lf in facts.lines:
        i = lf.index
        asked = any(k.startswith(f"line_{i}_") for k in probs)
        kind = _choice(probs, f"line_{i}_kind", LINE_KINDS)
        po_line = _argmax(probs, [(f"line_{i}_po_{pl.index}", pl.index) for pl in facts.po_lines])
        figure = _argmax(
            probs, [(f"line_{i}_evidence_{fg.index}", fg.index) for fg in facts.evidence_figures]
        )
        if kind == "work_or_goods":
            scope = _state(probs, f"line_{i}_scope", SCOPE_OPTIONS, asked, floor)
            completion = _state(probs, f"line_{i}_completion", COMPLETION_OPTIONS, asked, floor)
        elif (
            kind == "charge_on_top"
        ):
            scope = "component_also_billed" if yes(f"line_{i}_scope_component_also_billed") else "priced_unit"
            completion = "confirmed"
        else:  # Billing mechanics are excluded from scope and completion judgments.
            scope, completion = "priced_unit", "confirmed"
        lines.append(
            LineFinding(
                index=i,
                kind=kind,
                scope=scope,
                completion=completion,
                rebilled=yes(f"line_{i}_rebilled"),
                owner_declined=yes(f"line_{i}_owner_declined"),
                unexplained_fee=yes(f"line_{i}_unexplained_fee"),
                po_line=po_line,
                evidence_figure=figure,
                rate_differs=yes(f"line_{i}_rate_differs"),
            )
        )
    priors = [
        PriorFinding(index=pi.index, relation=_state(probs, f"prior_{pi.index}", PRIOR_OPTIONS, floor=floor))
        for pi in facts.prior_invoices
    ]
    approvals = [
        ApprovalFinding(
            index=a.index, reading=_state(probs, f"approval_{a.index}", APPROVAL_OPTIONS, floor=floor)
        )
        for a in facts.approvals_with_comment
    ]
    senders = [
        SenderFinding(index=s.index, identity=_state(probs, f"sender_{s.index}", SENDER_OPTIONS, floor=floor))
        for s in facts.unknown_senders
    ]
    basis = _choice(probs, "price_basis", (*PRICE_BASIS_OPTIONS, "none", "unsure")) or "none"
    retainage_choice = _choice(probs, "retainage", (*[str(int(lv)) for lv in RETAINAGE_LEVELS], "none"))
    retainage = None if retainage_choice in (None, "none") else float(retainage_choice)
    discount_choice = _choice(probs, "discount_days", (*[str(d) for d in DISCOUNT_DAY_LEVELS], "none"))
    discount = None if discount_choice in (None, "none") else int(discount_choice)

    return Findings(
        source="model",
        lines=lines,
        priors=priors,
        approvals=approvals,
        senders=senders,
        statement_not_invoice=yes("statement_not_invoice"),
        discount_days=discount,
        pressure_pattern=yes("unusual_urgency"),
        bank_change_claimed=yes("bank_change_claimed_in_comms"),
        different_entity=yes("different_entity"),
        price_basis=basis,
        billed_above_basis=yes("billed_above_basis") and basis not in ("none", "unsure"),
        variance_explained_by_split=clear("lines_restate_po_pricing"),
        retainage_pct=retainage,
        too_vague=yes("too_ambiguous"),
        adjustment_unexplained=yes("unexplained_charges"),
        adjustment_duplicates_line=yes("adjustment_duplicates_line"),
        tax_two_rates=yes("tax_two_rates"),
    )


def from_label(facts: ExactFacts, form: dict[str, Any]) -> Findings:
    """Read reference findings; null means the finding has not been supplied."""
    data = dict(form)
    lines_in = {int(x["index"]): x for x in data.pop("lines", [])}
    lines = []
    for lf in facts.lines:
        x = lines_in.get(lf.index, {})
        kind = x.get("kind")
        work = kind == "work_or_goods"
        charge = kind == "charge_on_top" and x.get("scope") == "component_also_billed"
        lines.append(
            LineFinding(
                index=lf.index,
                kind=kind,
                scope=(x.get("scope") or "priced_unit") if work or charge else "priced_unit",
                completion=(x.get("completion") or "confirmed") if work else "confirmed",
                rebilled=bool(x.get("rebilled", False)),
                owner_declined=bool(x.get("owner_declined", False)),
                unexplained_fee=bool(x.get("unexplained_fee", False)),
                po_line=x.get("po_line"),
                evidence_figure=x.get("evidence_figure"),
                rate_differs=bool(x.get("rate_differs", False)),
                why=str(x.get("why", "")),
            )
        )
    priors = [
        PriorFinding(index=int(x["index"]), relation=x.get("relation") or "unrelated")
        for x in data.pop("priors", [])
    ]
    approvals = [
        ApprovalFinding(index=int(x["index"]), reading=x.get("reading") or "approves_as_billed")
        for x in data.pop("approvals", [])
    ]
    senders = [
        SenderFinding(index=int(x["index"]), identity=x.get("identity") or "plausible")
        for x in data.pop("senders", [])
    ]
    for legacy in ("notes", "expected_actions", "v3"):
        data.pop(legacy, None)
    if data.get("price_basis") is None:
        data["price_basis"] = "none"
    return Findings(source="label", lines=lines, priors=priors, approvals=approvals, senders=senders, **data)


def unfilled(form: dict[str, Any]) -> list[str]:
    """Return state findings left null, the marker for an unanswered finding."""
    out = []
    for x in form.get("lines", []):
        if x.get("kind") is None:
            out.append(f"lines[{x['index']}].kind")
        if x.get("kind") == "work_or_goods":
            for k in ("scope", "completion"):
                if x.get(k) is None:
                    out.append(f"lines[{x['index']}].{k}")
    for key, field in (("priors", "relation"), ("approvals", "reading"), ("senders", "identity")):
        for x in form.get(key, []):
            if x.get(field) is None:
                out.append(f"{key}[{x['index']}].{field}")
    if form.get("price_basis") is None:
        out.append("price_basis")
    return out


def blank_form(facts: ExactFacts, packet: Any) -> dict[str, Any]:
    lines = [
        {
            "index": lf.index,
            "description": lf.description,
            "amount": lf.amount,
            "quantity": lf.quantity,
            "unit": lf.unit,
            "unit_price": lf.unit_price,
            "kind": None,
            "scope": None,
            "completion": None,
            "rebilled": False,
            "owner_declined": False,
            "unexplained_fee": False,
            "po_line": None,
            "evidence_figure": None,
            "rate_differs": False,
            "why": "",
        }
        for lf in facts.lines
    ]
    contract = packet.contract
    return {
        "lines": lines,
        "po_lines": [pl.model_dump() for pl in facts.po_lines],
        "evidence_figures": [fg.model_dump() for fg in facts.evidence_figures],
        "priors": [{**pi.model_dump(), "relation": None} for pi in facts.prior_invoices],
        "approvals": [{**a.model_dump(), "reading": None} for a in facts.approvals_with_comment],
        "senders": [{**s.model_dump(), "identity": None} for s in facts.unknown_senders],
        "payment_terms": packet.invoice.fields.payment_terms,
        "contract_excerpts": list(contract.relevant_excerpts) if contract else [],
        "statement_not_invoice": False,
        "discount_days": None,
        "pressure_pattern": False,
        "bank_change_claimed": False,
        "different_entity": False,
        "price_basis": None,
        "billed_above_basis": False,
        "variance_explained_by_split": False,
        "retainage_pct": None,
        "too_vague": False,
        "adjustment_unexplained": False,
        "adjustment_duplicates_line": False,
        "tax_two_rates": False,
        "notes": "",
        "expected_actions": None,
    }


FORM_ONLY_KEYS = (
    "po_lines",
    "evidence_figures",
    "payment_terms",
    "contract_excerpts",
    "notes",
    "expected_actions",
    "v3",
)
_PRIOR_KEEP = ("index", "relation")
_APPROVAL_KEEP = ("index", "reading")
_SENDER_KEEP = ("index", "identity")


def strip_form(form: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in form.items() if k not in FORM_ONLY_KEYS}
    out["lines"] = [
        {k: v for k, v in line.items() if k in ("index", *LINE_JUDGEMENT_KEYS, "why")}
        for line in out.get("lines", [])
    ]
    out["priors"] = [{k: v for k, v in x.items() if k in _PRIOR_KEEP} for x in out.get("priors", [])]
    out["approvals"] = [{k: v for k, v in x.items() if k in _APPROVAL_KEEP} for x in out.get("approvals", [])]
    out["senders"] = [{k: v for k, v in x.items() if k in _SENDER_KEEP} for x in out.get("senders", [])]
    return out
