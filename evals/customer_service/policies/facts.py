from __future__ import annotations

from ..gates import Fact

handed_off = Fact("a human already owns the conversation", lambda c: c.state.handed_off)
no_pending = Fact("no proposal is pending", lambda c: c.state.pending is None)
intent_unset = Fact("no intent has been set yet", lambda c: c.state.intent is None)


def pending_is(p: str) -> Fact:
    return Fact(f"the pending proposal is `{p}`", lambda c, p=p: c.state.pending == p)


def intent_is(i: str) -> Fact:
    return Fact(f"the recorded intent is `{i}`", lambda c, i=i: c.state.intent == i)


card_active = Fact("the card on file is active", lambda c: c.account.card_status == "active")
card_frozen = Fact("the card on file is already frozen", lambda c: c.account.card_status == "frozen")
has_transaction = Fact("a transaction is on file for this conversation", lambda c: c.account.transaction is not None)
no_transaction = Fact("no transaction is on file", lambda c: c.account.transaction is None)
has_refund_claim = Fact("a refund claim is on file", lambda c: c.account.refund is not None)
no_refund_claim = Fact("no refund claim is on file", lambda c: c.account.refund is None)
refund_eligible = Fact("the claim on file is eligible for a refund and not yet paid",
                       lambda c: c.account.refund is not None and c.account.refund.eligible and not c.account.refund.already_issued)
refund_ineligible = Fact("the claim on file is not eligible for a refund",
                         lambda c: c.account.refund is not None and not c.account.refund.eligible and not c.account.refund.already_issued)
refund_already_issued = Fact("the refund on file was already issued",
                             lambda c: c.account.refund is not None and c.account.refund.already_issued)


def refund_at_most(usd: float) -> Fact:
    return Fact(f"the refund amount on file is at most ${usd:g}", lambda c, u=usd: c.account.refund is not None and c.account.refund.amount_usd <= u)


def refund_over(usd: float) -> Fact:
    return Fact(f"the refund amount on file is over ${usd:g}", lambda c, u=usd: c.account.refund is not None and c.account.refund.amount_usd > u)


has_fee = Fact("an outstanding fee is on file", lambda c: c.account.outstanding_fee_usd > 0)
has_order = Fact("an order is on file", lambda c: c.account.order is not None)
no_order = Fact("no order is on file", lambda c: c.account.order is None)
subscription_active = Fact("the subscription is active", lambda c: c.account.subscription_active)


def tenure_at_least(months: int) -> Fact:
    return Fact(f"the customer's tenure is at least {months} months", lambda c, m=months: c.customer.tenure_months >= m)


def prior_refunds_under(n: int) -> Fact:
    return Fact(f"fewer than {n} refunds in the last 90 days", lambda c, n=n: c.customer.prior_refunds_90d < n)


vip = Fact("the customer is a VIP", lambda c: c.customer.vip)


def channel_is(*channels: str) -> Fact:
    return Fact(f"the transaction on file is {' or '.join(f'`{c}`' for c in channels)}",
                lambda c, ch=channels: c.account.transaction is not None and c.account.transaction.channel in ch)


refund_not_issued = Fact("the record shows no refund issued", lambda c: not (c.account.refund and c.account.refund.already_issued))
has_new_assistant_message = Fact("the assistant spoke since the previous customer message", lambda c: bool(c.new_assistant_message_numbers()))
