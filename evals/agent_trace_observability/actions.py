from __future__ import annotations

from enum import StrEnum


class State(StrEnum):
    """Task success crossed with user satisfaction."""

    HEALTHY = "healthy"  # worked, happy
    SILENT_FAILURE = "silent_failure"  # failed, happy
    OVERT_FAILURE = "overt_failure"  # failed, unhappy
    EXPECTATION_GAP = "expectation_gap"  # worked, unhappy


class Action(StrEnum):
    PAGE_ON_CALL = "page_on_call"
    PRIORITY_REVIEW = "priority_review"
    PRIORITY_REVIEW_NEW_EVAL_CASE = "priority_review_new_eval_case"
    FILE_ISSUE_ROUTE_TO_OWNER = "file_issue_route_to_owner"
    HUMAN_REVIEW = "human_review"
    COUNT_ONLY = "count_only"
    NOT_A_BUG = "not_a_bug"
    AUTO_CLOSE = "auto_close"


MEANING: dict[Action, str] = {
    Action.PAGE_ON_CALL: "security review, wake someone",
    Action.PRIORITY_REVIEW: "jump the human queue",
    Action.PRIORITY_REVIEW_NEW_EVAL_CASE: "jump the queue, and the trace becomes a regression case",
    Action.FILE_ISSUE_ROUTE_TO_OWNER: "new engineering defect, routed by attribution",
    Action.HUMAN_REVIEW: "ordinary queue",
    Action.COUNT_ONLY: "no human; increments an infrastructure rate that alerts in aggregate",
    Action.NOT_A_BUG: "goes to product, never to an engineering queue",
    Action.AUTO_CLOSE: "closes with no human -- the volume exit",
}

# Human attention cost, not compute cost.
COST: dict[Action, str] = {
    Action.PAGE_ON_CALL: "interrupt",
    Action.PRIORITY_REVIEW: "queue_front",
    Action.PRIORITY_REVIEW_NEW_EVAL_CASE: "queue_front",
    Action.FILE_ISSUE_ROUTE_TO_OWNER: "queue",
    Action.HUMAN_REVIEW: "queue",
    Action.COUNT_ONLY: "none",
    Action.NOT_A_BUG: "none",
    Action.AUTO_CLOSE: "none",
}

COSTS = ("interrupt", "queue_front", "queue", "none")

NEEDS_A_HUMAN = frozenset(d for d, cost in COST.items() if cost != "none")
