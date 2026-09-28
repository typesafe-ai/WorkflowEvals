from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.question_types import (
    Choice,
    NoulCriteria,
    Noul,
    Score,
)

from core.answers import NodeAnswers


DETECTOR_DOMAIN = {"EDR": "endpoint", "IDP": "identity", "NET": "network", "MAIL": "email"}


@dataclass(frozen=True)
class State:
    environment: str
    tier: int  # 0 = domain controller, 1 = server, 2 = workstation
    domain: str


def read_state(state: dict[str, Any]) -> State:
    asset = state["context"]["asset"]
    prefix = state["alert"].split("-", 1)[0].strip().upper()
    return State(
        environment=str(asset.get("environment", "dev")),
        tier=int(asset.get("tier", 2)),
        domain=DETECTOR_DOMAIN.get(prefix, "endpoint"),
    )


def _noul(instructions: str, true: str, false: str) -> Noul:
    return Noul(instructions=instructions, criteria=NoulCriteria(true=true, false=false))


EVIDENCE_LEVELS = [
    "Speculative: hedged language, or a single weak indicator",
    "Suggestive: one concrete indicator, uncorroborated",
    "Corroborated: independent indicators agree",
    "Confirmed: direct proof of unauthorized activity",
]

TRIAGE = {
    "is_true_positive": _noul(
        "Given the alert and its context records, does this describe unauthorized activity, "
        "as opposed to authorized activity that a detector flagged?",
        "Someone is doing something they were not authorized to do",
        "The activity was authorized, expected, or did not happen -- administrative work, a "
        "sanctioned tool, a test, or a detector firing on nothing",
    ),
    "context_explains_activity": _noul(
        "Do the context records -- tickets, registrations, schedules, authorization excerpts "
        "-- account for the flagged activity?",
        "A specific record covers this specific activity: the same actor, asset, or window, "
        "authorized in advance",
        "No record covers it, or the records that exist are about something adjacent. "
        "Silence is not an explanation",
    ),
    "evidence_strength": Score(
        instructions="How strong is the evidence that the activity is unauthorized?",
        criteria=EVIDENCE_LEVELS,
    ),
}

PREDICATES = {
    "credentials_exposed": (
        "Have credentials been disclosed to, or captured by, someone unauthorized?",
        "A password, key or token has been submitted to an attacker-controlled destination, "
        "or is otherwise in hands it should not be in",
        "No credential has left the control of its owner",
    ),
    "session_in_attacker_hands": (
        "Is a live session or token being used by someone unauthorized right now?",
        "An authenticated session or token is in active use by someone other than the owner",
        "Every active session belongs to the person it was issued to, or none are active",
    ),
    "malicious_content_in_mailboxes": (
        "Is malicious mail sitting in user inboxes at this moment?",
        "A malicious message is still delivered and reachable in one or more mailboxes",
        "Nothing malicious is in any mailbox, or the incident is not about mail",
    ),
    "attacker_persistence_present": (
        "Is there a mechanism that would survive a reboot and bring the activity back?",
        "A scheduled task, service, startup entry, key or implant placed by the attacker is "
        "still in place",
        "Nothing would restart the activity once it stopped",
    ),
    "malicious_process_running": (
        "Is a specific malicious process or task executing right now?",
        "A named process, script or scheduled task is running and doing the harm described",
        "Nothing is executing; the evidence is of something that already ran or never ran",
    ),
    "outbound_channel_active": (
        "Is data or command traffic leaving to a destination under attacker control?",
        "An outbound channel to an attacker destination is carrying traffic now",
        "No outbound channel is open, or the destination is legitimate",
    ),
    "attacker_modified_configuration": (
        "Did the attacker create or change a configuration that persists on its own?",
        "Mailbox forwarding rules, delegations, OAuth consents, keys or policy entries were "
        "created or altered during the incident",
        "No configuration was changed, or the changes were made by their legitimate owner",
    ),
    "activity_ongoing": (
        "Is the activity live or imminent, rather than finished?",
        "It is happening now, or the evidence points at an imminent next step",
        "It is finished, already remediated, or a historical record of something stopped",
    ),
    "spread_beyond_initial_entity": (
        "Has this reached beyond the entity that was originally flagged?",
        "A second host, account or mailbox is implicated, or lateral movement is underway",
        "Everything in evidence is confined to the entity that was flagged",
    ),
}

SCOPES = ("single_entity", "workgroup", "organization_wide")
ATTACK_TYPES = ("host_compromise", "account_takeover", "mail_campaign", "exfiltration")

CONTAINMENT = {
    **{key: _noul(*text) for key, text in PREDICATES.items()},
    "affected_scope": Choice(
        instructions="How far does this reach?",
        criteria={
            "single_entity": "One host, account or mailbox",
            "workgroup": "A team, a distribution list, or a handful of related assets",
            "organization_wide": "Everyone, or an asset the whole organization depends on",
        },
    ),
    "attack_type": Choice(
        instructions="What kind of attack does the evidence describe?",
        criteria={
            "host_compromise": "Malware, ransomware, or persistence on a host",
            "account_takeover": "Stolen or abused credentials or sessions",
            "mail_campaign": "Phishing or malicious mail sitting in user inboxes",
            "exfiltration": "Data staging or outbound theft in progress",
        },
    ),
}

PANELS = {"triage": TRIAGE, "containment": CONTAINMENT}


P_CLOSE = 0.15
CTX_MIN = 0.50
TIER_NEVER_CLOSE = 0

P_ACT = 0.75

NOTIFY_LOW, NOTIFY_HIGH = 0.15, 0.60

BANDS = ("close", "middle", "act")


def gate_1(a: NodeAnswers, state: State) -> str:
    p = a.is_true_positive.noul
    if p < P_CLOSE and a.context_explains_activity.noul > CTX_MIN and state.tier != TIER_NEVER_CLOSE:
        return "close"
    if p > P_ACT:
        return "act"
    return "middle"


def middle_label(a: NodeAnswers, state: State) -> str:
    p = a.is_true_positive.noul
    if state.domain == "identity" and NOTIFY_LOW < p <= NOTIFY_HIGH:
        return "notify_user"
    return "escalate_tier2"


CRED_REAUTH = 0.40
CRED_REVOKE = 0.60
SESSION_REVOKE = 0.50
SESSION_DISABLE = 0.65
PROCESS_KILL = 0.60
PERSISTENCE = 0.60
ISOLATE_ACTIVE_MIN = 0.50
MAIL_QUARANTINE = 0.50
MAIL_PURGE = 0.70
CONFIG_CHANGED = 0.60
EGRESS_ACTIVE = 0.45
SPREAD_MIN = 0.50


@dataclass(frozen=True)
class Rule:
    """All above thresholds and named facts must hold; any_above requires at least one match."""

    action: str
    above: tuple[tuple[str, float], ...] = ()
    any_above: tuple[tuple[str, float], ...] = ()
    facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Group:
    """Enter on any matching threshold; choose the first eligible action in the ladder."""

    name: str
    entered_on: tuple[tuple[str, float], ...]
    ladder: tuple[Rule, ...]


FACTS: dict[str, tuple[str, Callable[[NodeAnswers, State], bool]]] = {
    "scope_beyond_one": (
        "affected_scope is not single_entity",
        lambda a, s: a.affected_scope.choice != "single_entity",
    ),
    "scope_org_wide": (
        "affected_scope is organization_wide",
        lambda a, s: a.affected_scope.choice == "organization_wide",
    ),
    "is_exfiltration": (
        "attack_type is exfiltration",
        lambda a, s: a.attack_type.choice == "exfiltration",
    ),
    "is_mail_campaign": (
        "attack_type is mail_campaign",
        lambda a, s: a.attack_type.choice == "mail_campaign",
    ),
    "isolation_permitted": (
        f"(either the asset is not in production, or both `activity_ongoing` > {ISOLATE_ACTIVE_MIN} "
        f"and `spread_beyond_initial_entity` > {SPREAD_MIN})",
        lambda a, s: s.environment != "prod"
        or (a.activity_ongoing.noul > ISOLATE_ACTIVE_MIN and a.spread_beyond_initial_entity.noul > SPREAD_MIN),
    ),
}

# Order actions strongest first; stronger actions have stricter conditions.
PLAYBOOK: tuple[Group, ...] = (
    Group(
        "data is leaving now",
        entered_on=(("outbound_channel_active", EGRESS_ACTIVE),),
        ladder=(
            Rule("block_destination", above=(("outbound_channel_active", EGRESS_ACTIVE),), facts=("scope_org_wide",)),
            Rule("block_egress_asset", above=(("outbound_channel_active", EGRESS_ACTIVE),)),
        ),
    ),
    Group(
        "the attacker has access",
        entered_on=(("session_in_attacker_hands", SESSION_REVOKE), ("credentials_exposed", CRED_REAUTH)),
        ladder=(
            Rule(
                "disable_account",
                above=(("session_in_attacker_hands", SESSION_DISABLE), ("spread_beyond_initial_entity", SPREAD_MIN)),
            ),
            Rule("revoke_access_key", above=(("credentials_exposed", CRED_REVOKE),), facts=("is_exfiltration",)),
            Rule(
                "revoke_sessions",
                any_above=(("session_in_attacker_hands", SESSION_REVOKE), ("credentials_exposed", CRED_REVOKE)),
            ),
            Rule("require_reauth", above=(("credentials_exposed", CRED_REAUTH),)),
        ),
    ),
    Group(
        "malicious mail is delivered",
        entered_on=(("malicious_content_in_mailboxes", MAIL_QUARANTINE),),
        ladder=(
            Rule(
                "block_sender",
                above=(("malicious_content_in_mailboxes", MAIL_QUARANTINE),),
                facts=("is_mail_campaign", "scope_beyond_one"),
            ),
            Rule("purge_mailboxes", above=(("malicious_content_in_mailboxes", MAIL_PURGE),)),
            Rule("quarantine_message", above=(("malicious_content_in_mailboxes", MAIL_QUARANTINE),)),
        ),
    ),
    Group(
        "the host is compromised",
        entered_on=(("attacker_persistence_present", PERSISTENCE), ("malicious_process_running", PROCESS_KILL)),
        ladder=(
            Rule(
                "isolate_host",
                above=(("attacker_persistence_present", PERSISTENCE), ("activity_ongoing", ISOLATE_ACTIVE_MIN)),
                facts=("isolation_permitted",),
            ),
            Rule("kill_process", above=(("malicious_process_running", PROCESS_KILL),)),
            Rule("quarantine_file", above=(("attacker_persistence_present", PERSISTENCE),)),
        ),
    ),
    Group(
        "configuration was changed",
        entered_on=(("attacker_modified_configuration", CONFIG_CHANGED),),
        ladder=(Rule("remove_forwarding_rules", above=(("attacker_modified_configuration", CONFIG_CHANGED),)),),
    ),
)

EMPTY_ACT_FALLBACK = "escalate_urgent"

RULES: tuple[Rule, ...] = tuple(rule for group in PLAYBOOK for rule in group.ladder)


def holds(rule: Rule, a: NodeAnswers, state: State) -> bool:
    if not all(a[q].noul > t for q, t in rule.above):
        return False
    if rule.any_above and not any(a[q].noul > t for q, t in rule.any_above):
        return False
    return all(FACTS[name][1](a, state) for name in rule.facts)


def gate_2(a: NodeAnswers, state: State) -> str:
    for group in PLAYBOOK:
        if not any(a[q].noul > t for q, t in group.entered_on):
            continue
        for rule in group.ladder:
            if holds(rule, a, state):
                return rule.action
        # If a group has no eligible action, continue to the next group.
    return EMPTY_ACT_FALLBACK


LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    "close": ("auto_close",),
    "queue": ("escalate_tier2", "notify_user"),
    "page": ("escalate_urgent",),
    "light": (
        "require_reauth",
        "kill_process",
        "quarantine_file",
        "quarantine_message",
        "remove_forwarding_rules",
        "block_egress_asset",
    ),
    "heavy": (
        "revoke_sessions",
        "disable_account",
        "revoke_access_key",
        "isolate_host",
        "purge_mailboxes",
        "block_sender",
        "block_destination",
    ),
}
LABELS: tuple[str, ...] = tuple(label for group in LABEL_GROUPS.values() for label in group)


Judge = Callable[[dict[str, Any], str, dict[str, Any]], NodeAnswers]


@dataclass
class Trace:
    band: str
    label: str
    state: State
    triage: NodeAnswers | None = None
    containment: NodeAnswers | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "band": self.band,
            "label": self.label,
            "state": {"environment": self.state.environment, "tier": self.state.tier, "domain": self.state.domain},
            "triage": self.triage.to_json() if self.triage else None,
            "containment": self.containment.to_json() if self.containment else None,
        }


def run(input_state: dict[str, Any], judge: Judge) -> Trace:
    state = read_state(input_state)
    triage = judge(input_state, "triage", TRIAGE)
    band = gate_1(triage, state)
    if band == "close":
        return Trace(band, "auto_close", state, triage)
    if band == "middle":
        return Trace(band, middle_label(triage, state), state, triage)
    containment = judge(input_state, "containment", CONTAINMENT)
    return Trace(band, gate_2(containment, state), state, triage, containment)
