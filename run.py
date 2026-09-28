#!/usr/bin/env python3
"""Run a workflow and score its decisions against reference answers."""

import argparse
import json
from pathlib import Path

from core.io import load_results, write_record
from core.pricing import load_pricing
from core.providers import configuration
from core.runner import execute, plan
from core.scoring import add_reference, build_score_data
from core.workflow import WORKFLOWS, load


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=WORKFLOWS)
    parser.add_argument(
        "--model", default="typesafe:jev-1.13.0", help="provider:model (default: %(default)s)"
    )
    parser.add_argument("--name", help="Run name (default: model name); reuse to resume")
    parser.add_argument(
        "--thinking", choices=("default", "off", "minimal", "low", "medium", "high", "xhigh", "max")
    )
    parser.add_argument("--limit", type=int, help="Run the first N cases in case-ID order")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent cases (default: 4)")
    parser.add_argument(
        "--questions", choices=("one", "all"), help="Default: one per LLM request, all for TypeSafe"
    )
    parser.add_argument(
        "--question-workers", type=int, default=8, help="Concurrent questions per case (default: 8)"
    )
    parser.add_argument("--policy", action="append", help="Select a policy; repeat for several")
    parser.add_argument("--dataset-revision", help="Hugging Face revision (default: latest main)")
    parser.add_argument(
        "--reference", type=Path, action="append", default=[], help="Saved run to use as reference answers"
    )
    parser.add_argument("--base-url", help="TypeSafe endpoint override")
    parser.add_argument("--timeout", type=float, default=600, help="Request timeout in seconds")
    parser.add_argument("--max-output-tokens", type=int, default=64000)
    parser.add_argument("--retry-errors", action="store_true", help="Rerun failed cases")
    parser.add_argument(
        "--dry-run", action="store_true", help="Download and check inputs without model calls or run outputs"
    )
    args = parser.parse_args(argv)

    try:
        if args.workers < 1 or args.question_workers < 1:
            raise ValueError("Workers must be positive")
        config = configuration(
            args.model,
            args.thinking,
            base_url=args.base_url,
            timeout_s=args.timeout,
            max_output_tokens=args.max_output_tokens,
            question_mode=args.questions,
        )
        name = args.name if args.name is not None else config.model.rsplit("/", 1)[-1]
        if not name.strip() or name in (".", "..") or Path(name).name != name or "\\" in name:
            raise ValueError("Run name must be a single folder name, not a path")
        output = Path(__file__).resolve().parent / "runs" / args.workflow / name
        workflow = load(args.workflow, args.dataset_revision)
        for reference in args.reference:
            add_reference(workflow.bundle, load_results(reference))
        job = plan(workflow, config, policy_ids=args.policy, limit=args.limit)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "workflow": args.workflow,
                    "cases": len(job.case_ids),
                    "policies": job.policy_ids,
                    "model": config.model_dump(mode="json"),
                    "dataset": workflow.bundle.dataset_source,
                    "name": name,
                    "output": str(output),
                },
                indent=2,
            )
        )
        return 0

    results = execute(
        workflow,
        job,
        output,
        run_id=name,
        prices=load_pricing(),
        workers=args.workers,
        question_workers=args.question_workers,
        retry_errors=args.retry_errors,
        progress=lambda message: print(message, flush=True),
    )
    scores = build_score_data(results, policy_ids=job.policy_ids)
    write_record(output / "scores.json", scores)
    for series in scores.series:
        print(f"{series.label_set_id}: {json.dumps(series.metrics)} ({series.decisions} scored decisions)")
    if not scores.series:
        print(
            "No eligible reference labels; use --reference with a run from another model to score these decisions."
        )
    print(f"Saved {output / 'results.json'} and {output / 'scores.json'}")
    selected = set(job.case_ids)
    cases = [case for case in results.runs[0].cases if case.case_id in selected]
    costs = [case.execution.cost.usd for case in cases if case.execution and case.execution.cost is not None]
    times = [
        case.execution.wall_time_s
        for case in cases
        if case.execution and case.execution.wall_time_s is not None
    ]
    count = len(cases)
    average_cost = f"${sum(costs) / count:.6f}" if count and len(costs) == count else "unknown"
    average_time = f"{sum(times) / count:.2f}s" if count and len(times) == count else "unknown"
    print(f"Average per case ({count} cases): estimated cost={average_cost} | elapsed time={average_time}")
    return int(any(case.execution and case.execution.status != "ok" for case in results.runs[0].cases))


if __name__ == "__main__":
    raise SystemExit(main())
