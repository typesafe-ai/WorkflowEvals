#!/usr/bin/env python3
"""Export a comparison of saved workflow runs as PNG, SVG, or PDF."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from core.contract import CaseResult, Cost, Decision, Execution, Metric, ModelIdentity, Results, Run
from core.dataset import DATASETS, action
from core.io import load_results
from core.scoring import build_score_data
from core.workflow import WORKFLOWS, load

PROVIDER_COLORS = {
    "typesafe": "#E551BA",
    "openai": "#09AEA1",
    "anthropic": "#E8912A",
    "fireworks": "#8A7BF7",
    "groq": "#C4CFCE",
    "cerebras": "#8E9C9B",
}
UNKNOWN_COLOR = "#ABBAB9"
METRICS = {
    "exact": ("exact_actions", "Exact action-set agreement"),
    "primary": ("primary_action", "Primary action agreement"),
}
AXES = {
    "cost": ("mean_cost_usd", "Mean cost per case (USD, log scale)"),
    "time": ("mean_wall_time_s", "Mean elapsed time per case (seconds, log scale)"),
}


def published_runs(workflow: str) -> Results:
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    bundle = load(workflow).bundle
    repo = DATASETS[workflow]
    revision = bundle.dataset_source.rsplit("/", 1)[-1]
    manifest = json.loads(
        Path(hf_hub_download(repo, "dataset.json", repo_type="dataset", revision=revision)).read_text()
    )
    try:
        path = hf_hub_download(repo, "data/run_results.parquet", repo_type="dataset", revision=revision)
    except RemoteEntryNotFoundError as exc:
        raise ValueError(f"No published run results in {bundle.dataset_source}") from exc
    rows = parquet.read_table(
        path,
        columns=[
            "run_id",
            "model",
            "case_id",
            "status",
            "cost_usd",
            "wall_time_s",
            "summed_call_time_s",
            "input_tokens",
            "output_tokens",
            "calls",
            "decisions",
        ],
    ).to_pylist()
    hashes = {case.case_id: case.content_hash for case in bundle.cases}
    policies = {policy.policy_id: policy.revision for policy in bundle.policies}
    runs: dict[str, Run] = {}
    for row in rows:
        run_id, model = row["run_id"], row["model"]
        if run_id not in runs:
            provider = model["provider"]
            runs[run_id] = Run(
                run_id=f"hf:{run_id}",
                model=ModelIdentity(
                    name=f"{provider}:{model['name']}",
                    lab=provider if provider in ("openai", "anthropic", "typesafe") else "other",
                    effort=model["reasoning_effort"],
                ),
                configuration={"model": model, "hf_run_id": run_id, "dataset_source": bundle.dataset_source},
            )
        if model != runs[run_id].configuration["model"]:
            raise ValueError(f"Published run {run_id} contains inconsistent model settings")
        if row["case_id"] not in hashes:
            raise ValueError(f"Published run {run_id} contains an unknown case: {row['case_id']}")
        decisions = [
            Decision(
                policy_id=d["policy_id"],
                policy_revision=policies[d["policy_id"]],
                status=d["status"],
                actions=[action(a) for a in d["actions"]],
                primary_action=action(d["primary_action"]) if d["primary_action"] else None,
            )
            for d in row["decisions"]
        ]
        if not decisions and row["status"] == "error":
            decisions = [Decision.failed(p, rev, "Published execution failed") for p, rev in policies.items()]
        cost = None
        if row["cost_usd"] is not None:
            if manifest["runs"][run_id]["cost_bases"] != ["estimated_uncached"]:
                raise ValueError(f"Unsupported published cost basis for {run_id}")
            cost = Cost(usd=row["cost_usd"], basis="estimated_uncached")
        runs[run_id].cases.append(
            CaseResult(
                case_id=row["case_id"],
                case_hash=hashes[row["case_id"]],
                input_status="bound_annotation",
                source=f"{bundle.dataset_source}#{run_id}/{row['case_id']}",
                decisions=decisions,
                execution=Execution(
                    policy_ids=[d.policy_id for d in decisions],
                    policy_independent=True,
                    status=row["status"],
                    cost=cost,
                    wall_time_s=row["wall_time_s"],
                    summed_call_time_s=row["summed_call_time_s"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    calls=row["calls"],
                ),
            )
        )
    if not runs:
        raise ValueError(f"No published runs in {bundle.dataset_source}")
    bundle.runs = list(runs.values())
    return Results.model_validate(bundle.model_dump(mode="json"))


def combine(paths: list[Path], reference: str, published: Results | None = None) -> Results:
    bundles = [load_results(path) for path in paths]
    sources = [str(path) for path in paths]
    if published is not None:
        bundles.append(published)
        sources.append(published.dataset_source)
    bundle = bundles[0].model_copy(deep=True)
    label_id = bundle.references.get(reference, reference)
    label = next((label for label in bundle.label_sets if label.label_set_id == label_id), None)
    if label is None:
        raise ValueError(f"Reference {reference!r} is absent from {sources[0]}")
    for path, other in zip(sources[1:], bundles[1:]):
        for field in ("eval_id", "dataset_revision", "policies", "metrics"):
            if getattr(other, field) != getattr(bundle, field):
                raise ValueError(f"{path} has different {field}; compare runs of the same workflow version")
        if sorted(other.cases, key=lambda c: c.case_id) != sorted(bundle.cases, key=lambda c: c.case_id):
            raise ValueError(f"{path} has different input cases")
        other_label = next((item for item in other.label_sets if item.label_set_id == label_id), None)
        if other_label is None or other_label.revision != label.revision:
            raise ValueError(f"{path} has different {reference} reference labels")
    bundle.label_sets = [label]
    bundle.references = {reference: label_id}
    bundle.runs = [run for item in bundles for run in item.runs]
    return Results.model_validate(bundle.model_dump(mode="json"))


@dataclass
class Point:
    run_id: str
    model: ModelIdentity
    question_mode: str | None
    x: float | None
    y: float | None


def workflow_points(bundle: Results, metric: str, x_axis: str, policies: list[str] | None) -> list[Point]:
    comparison, y_title = METRICS[metric]
    field, _ = AXES[x_axis]
    bundle = bundle.model_copy(
        update={"metrics": [Metric(metric_id=comparison, title=y_title, comparison=comparison)]}
    )
    scores = build_score_data(bundle, policy_ids=policies, model_selection="all")
    runs = {run.run_id: run for run in bundle.runs}
    return [
        Point(
            run_id=s.run_id,
            model=runs[s.run_id].model,
            question_mode=runs[s.run_id].configuration.get("model", {}).get("question_mode"),
            x=getattr(s, field),
            y=s.metrics.get(comparison),
        )
        for s in scores.series
    ]


def average_points(workflows: dict[str, list[Point]], metric: str) -> list[Point]:
    groups: dict[tuple[str, str | None, str | None], dict[str, list[Point]]] = {}
    for workflow, points in workflows.items():
        if not any(point.y is not None for point in points):
            raise ValueError(
                f"No {metric} scores for {workflow}; an average requires the metric in every workflow"
            )
        for point in points:
            key = (point.model.name, point.model.effort, point.question_mode)
            groups.setdefault(key, {}).setdefault(workflow, []).append(point)

    def mean(values):
        return fmean(values) if values and all(value is not None for value in values) else None

    averaged = []
    for key, members in groups.items():
        missing = workflows.keys() - members.keys()
        if missing:
            print(f"Skipped {key}: missing workflows: {', '.join(sorted(missing))}")
            continue
        first = next(iter(members.values()))[0]
        averaged.append(
            Point(
                run_id=first.model.name,
                model=first.model,
                question_mode=first.question_mode,
                x=mean([mean([p.x for p in points]) for points in members.values()]),
                y=mean([mean([p.y for p in points]) for points in members.values()]),
            )
        )
    return averaged


def export(points: list[Point], title: str, metric: str, x_axis: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    from matplotlib.transforms import Bbox

    _, y_title = METRICS[metric]
    _, x_title = AXES[x_axis]
    series = []
    for item in points:
        if item.y is None:
            print(f"Skipped {item.run_id}: {metric} score is unavailable")
            continue
        if item.x is None or item.x <= 0:
            print(f"Skipped {item.run_id}: {x_axis} must be known and positive for the logarithmic axis")
            continue
        series.append(item)
    if not series:
        raise ValueError("No plottable runs for this metric, x axis, reference, and policy selection")
    providers = [s.model.name.partition(":")[0] for s in series]
    style = {
        "figure.facecolor": "#1E1E1E",
        "axes.facecolor": "#1E1E1E",
        "text.color": "#FEFEFE",
        "axes.labelcolor": "#D9DFDE",
        "xtick.color": "#ABBAB9",
        "ytick.color": "#FEFEFE",
        "font.size": 10,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 5))
        for provider in dict.fromkeys(providers):
            points = [s for s, p in zip(series, providers) if p == provider]
            ax.scatter(
                [s.x for s in points],
                [s.y * 100 for s in points],
                color=PROVIDER_COLORS.get(provider, UNKNOWN_COLOR),
                marker="D",
                s=55,
                label="TypeSafe" if provider == "typesafe" else provider,
                zorder=3,
                clip_on=False,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}" if x_axis == "cost" else f"{v:g}"))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        values = [s.y * 100 for s in series]
        ax.set_ylim(max(0, min(values) - 5), min(100, max(values) + 5))
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_title(title, pad=16)
        ax.set_axisbelow(True)
        ax.grid(color="#353737", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#515A59")
        ax.tick_params(axis="both", which="both", length=0)
        fig.legend(loc="lower center", ncol=min(6, len(set(providers))), frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        coordinates = [(s.x, s.y * 100) for s in series]
        radius = 5 * fig.dpi / 72
        occupied = [
            Bbox.from_bounds(x - radius, y - radius, 2 * radius, 2 * radius)
            for x, y in ax.transData.transform(coordinates)
        ]
        for item, provider, xy in zip(series, providers, coordinates):
            model = item.model.name.split(":", 1)[-1].rsplit("/", 1)[-1]
            label = ax.annotate(
                model,
                xy,
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8,
                color=PROVIDER_COLORS.get(provider, UNKNOWN_COLOR),
            )
            for dx, dy in ((7, 7), (7, -7), (-7, 7), (-7, -7)):
                label.set_position((dx, dy))
                label.set_ha("left" if dx > 0 else "right")
                label.set_va("bottom" if dy > 0 else "top")
                bounds = label.get_window_extent(renderer).padded(2)
                if not any(bounds.overlaps(other) for other in occupied):
                    break
            occupied.append(bounds)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=(*WORKFLOWS, "average"))
    parser.add_argument("--names", nargs="+", help="Local run names to include (default: all saved runs)")
    parser.add_argument(
        "--hf", action="store_true", help="Include published runs from the latest Hugging Face dataset"
    )
    parser.add_argument("--reference", default="consensus", help="Reference name or label-set ID")
    parser.add_argument(
        "--metric", choices=METRICS, default="exact", help="Agreement metric (default: exact)"
    )
    parser.add_argument("--x-axis", choices=AXES, default="cost", help="Horizontal axis (default: cost)")
    parser.add_argument("--policy", action="append", help="Select a policy; repeat for several")
    parser.add_argument("--out", type=Path, help="PNG, SVG, or PDF (default: runs/<workflow>/comparison.svg)")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent / "runs" / args.workflow
    try:
        if args.workflow == "average" and args.policy:
            raise ValueError("--policy applies to individual workflows, not average")
        if args.names and any(
            not n.strip() or n in (".", "..") or Path(n).name != n or "\\" in n for n in args.names
        ):
            raise ValueError("Run names must be folder names, not paths")
        output = args.out or root / "comparison.svg"
        if output.suffix.lower() not in (".png", ".svg", ".pdf"):
            raise ValueError("Choose a .png, .svg, or .pdf output")
        workflows = WORKFLOWS if args.workflow == "average" else (args.workflow,)
        points_by_workflow = {}
        for workflow in workflows:
            directory = root.parent / workflow
            paths = (
                [directory / name for name in args.names]
                if args.names
                else sorted(p.parent for p in directory.glob("*/results.json"))
            )
            if not paths and not args.hf:
                raise ValueError(f"No saved runs in {directory}; run run.py first or use --hf")
            published = published_runs(workflow) if args.hf else None
            if published is not None:
                print(f"Loaded {len(published.runs)} HF runs from {published.dataset_source}")
            bundle = combine(paths, args.reference, published)
            if bundle.eval_id != workflow:
                raise ValueError(f"Saved runs in {directory} belong to a different workflow")
            points_by_workflow[workflow] = workflow_points(bundle, args.metric, args.x_axis, args.policy)
        if args.workflow == "average":
            points = average_points(points_by_workflow, args.metric)
            title = "Average across workflows"
        else:
            points = points_by_workflow[args.workflow]
            title = bundle.title
        export(points, title, args.metric, args.x_axis, output)
    except ImportError as exc:
        parser.error(f"Install plotting dependencies with `uv sync --extra plot`: {exc}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
