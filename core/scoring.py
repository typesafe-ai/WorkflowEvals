from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import Field

from .contract import Decision, LabelSet, Metric, Record, Results, Run, fingerprint


def add_reference(results: Results, reference: Results) -> None:
    for field in ("eval_id", "dataset_revision", "cases", "policies", "metrics"):
        if getattr(results, field) != getattr(reference, field):
            raise ValueError(f"Reference has different {field}")
    if len(reference.runs) != 1 or not reference.runs[0].cases:
        raise ValueError("Reference must contain one nonempty run")
    run = reference.runs[0]
    label_id = f"reference-{run.run_id}"
    if any(label.label_set_id == label_id for label in results.label_sets):
        raise ValueError(f"Duplicate reference: {label_id}")
    cases = [case.model_copy(deep=True) for case in run.cases]
    for case in cases:
        case.execution, case.calls, case.details = None, [], {}
    results.label_sets.append(
        LabelSet(
            label_set_id=label_id,
            title=run.model.name,
            revision=fingerprint([case.model_dump(mode="json") for case in cases]),
            method="independent",
            sources=[run.model],
            cases=cases,
        )
    )


class Score(Record):
    run_id: str
    label_set_id: str
    case_id: str
    policy_id: str
    status: Literal[
        "scored",
        "prediction_error",
        "missing_prediction",
        "missing_label",
        "unresolved_label",
        "stale_input",
        "unverified_input",
        "policy_mismatch",
    ]
    unverified: bool = False
    metrics: dict[str, bool | None] = Field(default_factory=dict)


def compare(metric: Metric, actual: Decision, expected: Decision) -> bool | None:
    if metric.comparison == "exact_actions":
        return {a.key() for a in actual.actions} == {a.key() for a in expected.actions}
    if actual.primary_action is None or expected.primary_action is None:
        return None
    return actual.primary_action.key() == expected.primary_action.key()


def score_pair(
    bundle: Results,
    run: Run,
    label: LabelSet,
    *,
    allow_unverified: bool = False,
) -> list[Score]:
    actual_cases = {c.case_id: c for c in run.cases}
    expected_cases = {c.case_id: c for c in label.cases}
    hashes = {c.case_id: c.content_hash for c in bundle.cases}
    rows = []
    for case in bundle.cases:
        a, e = actual_cases.get(case.case_id), expected_cases.get(case.case_id)
        for policy in bundle.policies:
            ad = next((d for d in a.decisions if d.policy_id == policy.policy_id), None) if a else None
            ed = next((d for d in e.decisions if d.policy_id == policy.policy_id), None) if e else None
            unverified = any(c and c.input_status == "unverified" for c in (a, e))
            unverified = bool(
                unverified or (ad and not ad.policy_revision) or (ed and not ed.policy_revision)
            )
            status = "scored"
            if any(
                c and (c.input_status == "stale" or (c.case_hash and c.case_hash != hashes[case.case_id]))
                for c in (a, e)
            ):
                status = "stale_input"
            elif ed is None or ed.status in ("missing", "error"):
                status = "missing_label"
            elif ed.status == "unresolved":
                status = "unresolved_label"
            elif ad is None or ad.status == "missing":
                status = "missing_prediction"
            elif ad.policy_revision and ed.policy_revision and ad.policy_revision != ed.policy_revision:
                status = "policy_mismatch"
            elif unverified and not allow_unverified:
                status = "unverified_input"
            elif ad.status != "ok":
                status = "prediction_error"
            row = Score(
                run_id=run.run_id,
                label_set_id=label.label_set_id,
                case_id=case.case_id,
                policy_id=policy.policy_id,
                status=status,
                unverified=unverified,
            )
            if status == "scored":
                row.metrics = {m.metric_id: compare(m, ad, ed) for m in bundle.metrics}
            elif status == "prediction_error":
                row.metrics = {m.metric_id: False for m in bundle.metrics}
            rows.append(row)
    return rows


class Series(Record):
    run_id: str
    label_set_id: str
    decisions: int
    cases: int
    status_counts: dict[str, int]
    unverified_decisions: int
    metrics: dict[str, float | None]
    metric_denominators: dict[str, int]
    mean_cost_usd: float | None
    cost_cases: int
    cost_bases: list[str]
    mean_wall_time_s: float | None
    wall_time_cases: int
    mean_summed_call_time_s: float | None
    summed_call_time_cases: int


class ScoreData(Record):
    schema_version: Literal["1.3"] = "1.3"
    eval_id: str
    dataset_revision: str
    # Each reference has its own shared case/policy cohort and denominators.
    cohorts: dict[str, dict[str, list[str]]]
    cohort_ids: dict[str, str]
    policy_ids: list[str]
    allow_unverified: bool
    model_selection: Literal["exclude_source_model", "cross_lab", "all"]
    failure_handling: Literal["count_prediction_errors_as_wrong"] = "count_prediction_errors_as_wrong"
    series: list[Series]
    scores: list[Score]


def included(run: Run, label: LabelSet, selection: str) -> bool:
    if selection == "all":
        return True

    def name(model):
        return model.name.split(":", 1)[-1]

    if run.run_id in label.source_run_ids or name(run.model) in {name(m) for m in label.sources}:
        return False
    return selection != "cross_lab" or run.model.lab not in {m.lab for m in label.sources}


def build_score_data(
    bundle: Results,
    *,
    run_ids: list[str] | None = None,
    label_ids: list[str] | None = None,
    policy_ids: list[str] | None = None,
    allow_unverified: bool = False,
    model_selection: Literal["exclude_source_model", "cross_lab", "all"] = "exclude_source_model",
) -> ScoreData:
    """Compare runs on a shared case/policy cohort within each reference.

    Failed predictions count as wrong; missing or invalid data stays in coverage counts.
    Cost and timing cover the requested policy workload without apportioning shared calls."""

    def select(items, ids, key):
        if ids is None:
            return items
        known = {getattr(x, key) for x in items}
        if set(ids) - known:
            raise ValueError(f"Unknown {key}: {sorted(set(ids) - known)}")
        return [x for x in items if getattr(x, key) in ids]

    runs = select(bundle.runs, run_ids, "run_id")
    labels = select(bundle.label_sets, label_ids, "label_set_id")
    policies = select(bundle.policies, policy_ids, "policy_id")
    scope = {p.policy_id for p in policies}
    # Exclude runs that did not request every selected policy.
    runs = [r for r in runs if scope <= {d.policy_id for c in r.cases for d in c.decisions}]
    pairs = [(r, label) for label in labels for r in runs if included(r, label, model_selection)]
    rows_by_pair = {
        (r.run_id, label.label_set_id): [
            s for s in score_pair(bundle, r, label, allow_unverified=allow_unverified) if s.policy_id in scope
        ]
        for r, label in pairs
    }
    cohorts: dict[str, dict[str, list[str]]] = {}
    commons: dict[str, set[tuple[str, str]]] = {}
    cost_ids_by_label: dict[str, set[str]] = {}
    for label in labels:
        valid = [
            {(s.case_id, s.policy_id) for s in rows if s.status in ("scored", "prediction_error")}
            for (_, lid), rows in rows_by_pair.items()
            if lid == label.label_set_id
        ]
        common = set.intersection(*valid) if valid else set()
        cohort = {p.policy_id: sorted(cid for cid, pid in common if pid == p.policy_id) for p in policies}
        commons[label.label_set_id], cohorts[label.label_set_id] = common, cohort
        # Pooled cost and timing require the same cases across all requested policies.
        cost_ids_by_label[label.label_set_id] = (
            set.intersection(*(set(ids) for ids in cohort.values())) if cohort else set()
        )
    series = []
    for run, label in pairs:
        common, cost_ids = commons[label.label_set_id], cost_ids_by_label[label.label_set_id]
        all_rows = rows_by_pair[run.run_id, label.label_set_id]
        rows = [s for s in all_rows if (s.case_id, s.policy_id) in common]
        measurements = [
            c.execution
            for c in run.cases
            if c.case_id in cost_ids
            and c.execution
            and (
                set(c.execution.policy_ids) == scope
                or c.execution.policy_independent
                and scope <= set(c.execution.policy_ids)
            )
        ]
        costs = [m.cost.usd for m in measurements if m.cost is not None]
        wall = [m.wall_time_s for m in measurements if m.wall_time_s is not None]
        summed = [m.summed_call_time_s for m in measurements if m.summed_call_time_s is not None]
        # Report means only when every successful case has a measurement; missing values are not zero.
        by_case = {c.case_id: c for c in run.cases}
        needed = sum(
            1
            for cid in cost_ids
            if by_case.get(cid) and by_case[cid].execution and by_case[cid].execution.status == "ok"
        )

        def complete_mean(values, needed=needed):  # noqa: B008 -- bound per series on purpose
            return sum(values) / len(values) if values and len(values) >= needed else None

        values = {
            m.metric_id: [s.metrics[m.metric_id] for s in rows if s.metrics.get(m.metric_id) is not None]
            for m in bundle.metrics
        }
        series.append(
            Series(
                run_id=run.run_id,
                label_set_id=label.label_set_id,
                decisions=len(rows),
                cases=len({s.case_id for s in rows}),
                status_counts=dict(Counter(s.status for s in all_rows)),
                unverified_decisions=sum(s.unverified for s in rows),
                metrics={k: sum(v) / len(v) if v else None for k, v in values.items()},
                metric_denominators={k: len(v) for k, v in values.items()},
                mean_cost_usd=complete_mean(costs),
                cost_cases=len(costs),
                cost_bases=sorted({m.cost.basis for m in measurements if m.cost}),
                mean_wall_time_s=complete_mean(wall),
                wall_time_cases=len(wall),
                mean_summed_call_time_s=complete_mean(summed),
                summed_call_time_cases=len(summed),
            )
        )
    return ScoreData(
        eval_id=bundle.eval_id,
        dataset_revision=bundle.dataset_revision,
        cohorts=cohorts,
        cohort_ids={
            lid: fingerprint({"dataset": bundle.dataset_revision, "cohort": cohort})
            for lid, cohort in cohorts.items()
        },
        policy_ids=[p.policy_id for p in policies],
        allow_unverified=allow_unverified,
        model_selection=model_selection,
        series=series,
        scores=[s for rows in rows_by_pair.values() for s in rows],
    )
