"""Run a node whenever any policy profile needs its answers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from . import state as S
from core.answers import NodeAnswers
from core.answers import from_json as from_json_answers
from .actions import State
from .facts import Facts, read
from .gates import PROFILES, Breach, Profile, Verdict, classify, select_action, safety_breach
from .questions import CLARITY, OUTCOME, REACTION, SAFETY, QuestionSet, attribution_questions
from .trace import Run, ToolCall

Asker = Callable[[str, dict[str, Any], QuestionSet], NodeAnswers]


@dataclass
class NodeRun:
    node: str
    ran: bool
    reason: str
    n_questions: int = 0
    state: dict[str, Any] | None = None
    answers: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "ran": self.ran,
            "reason": self.reason,
            "n_questions": self.n_questions,
            "answers": self.answers,
            "state": self.state,
        }


@dataclass
class Result:
    facts: Facts
    verdicts: dict[str, Verdict]
    nodes: list[NodeRun] = field(default_factory=list)

    def to_details(self) -> dict[str, Any]:
        return {
            "facts": self.facts.to_json(),
            "nodes": [n.to_json() for n in self.nodes],
            "verdicts": {
                name: {
                    "action": v.action.value,
                    **v.to_outputs(),
                    "breach": None
                    if v.breach is None
                    else {"step": v.breach.call.label, "reason": v.breach.reason},
                }
                for name, v in self.verdicts.items()
            },
        }


def run_workflow(ask: Asker, run: Run, profiles: dict[str, Profile] = PROFILES) -> Result:
    facts = read(run)
    nodes: list[NodeRun] = []

    def gather(node: str, state: dict[str, Any], qs: QuestionSet, reason: str) -> NodeAnswers:
        answers = ask(node, state, qs)
        nodes.append(NodeRun(node, True, reason, len(qs), state, answers.to_json()))
        return answers

    def skip(node: str, reason: str) -> None:
        nodes.append(NodeRun(node, False, reason))

    safety: list[tuple[Any, NodeAnswers]] = []
    if facts.consequential_calls:
        for call in facts.consequential_calls:
            spec, grant = run.spec_for(call), run.grant_for(call)
            why = []
            if spec and spec.destructive:
                why.append("annotated destructive")
            if grant and grant.conditioned:
                why.append("conditioned by the charter")
            answers = gather(
                f"safety_{call.ordinal}",
                S.safety_state(run, call),
                SAFETY,
                f"{call.label} is {' and '.join(why)}",
            )
            safety.append((call, answers))
    else:
        skip("safety", "no destructive or charter-conditioned tool was called")

    outcome = gather("outcome", S.outcome_state(run), OUTCOME, "always")
    reaction: NodeAnswers | None = None
    if facts.has_reaction:
        reaction = gather(
            "reaction", S.reaction_state(run), REACTION, f"feedback channel: {facts.reaction_signal}"
        )
    else:
        skip("reaction", "no reaction on record; the satisfaction axis has no reading")

    states = {name: classify(outcome, reaction, profile).state for name, profile in profiles.items()}
    failed = {n for n, s in states.items() if s in (State.SILENT_FAILURE, State.OVERT_FAILURE)}
    gapped = {n for n, s in states.items() if s is State.EXPECTATION_GAP}

    if failed or facts.had_failure:
        why = f"read as failed under {sorted(failed)}" if failed else "a tool call failed"
        gather("attribution", S.attribution_state(run), attribution_questions(S.step_labels(run)), why)
    else:
        skip("attribution", "no profile reads the run as failed and no tool call failed")

    if gapped:
        gather(
            "clarity", S.clarity_state(run), CLARITY, f"read as an expectation gap under {sorted(gapped)}"
        )
    else:
        skip("clarity", "no profile reads the run as an expectation gap")

    answered = {n.node: NodeAnswers(from_json_answers(n.answers)) for n in nodes if n.ran}
    verdicts = decide(run, facts, answered, profiles)
    return Result(facts=facts, verdicts=verdicts, nodes=nodes)


def decide(
    run: Run,
    facts: Facts,
    answered: Mapping[str, NodeAnswers],
    profiles: dict[str, Profile] = PROFILES,
) -> dict[str, Verdict]:
    """Apply current policies to live or saved node answers."""
    safety: list[tuple[ToolCall, NodeAnswers]] = [
        (call, answered[f"safety_{call.ordinal}"])
        for call in facts.consequential_calls
        if f"safety_{call.ordinal}" in answered
    ]
    verdicts: dict[str, Verdict] = {}
    for name, profile in profiles.items():
        breaches: list[Breach] = []
        for call, answers in safety:
            breach = safety_breach(answers, call, run.grant_for(call), profile)
            if breach is not None:
                breaches.append(breach)
        verdicts[name] = select_action(
            profile=profile,
            facts=facts,
            agent_name=run.charter.agent_name,
            outcome=answered["outcome"],
            reaction=answered.get("reaction"),
            breaches=breaches,
            attribution=answered.get("attribution"),
            clarity=answered.get("clarity"),
        )
    return verdicts
