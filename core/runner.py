from __future__ import annotations

import fcntl
import importlib.metadata
import json
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from .contract import CaseResult, Decision, Record, Results, Run, describe, fingerprint
from .io import write_record, write_text
from .providers import ModelConfig, make_client, resolve_model
from .session import QuestionClient, Session
from .workflow import Workflow, revision


class Plan(Record):

    model: ModelConfig
    policy_ids: list[str]
    case_ids: list[str]


def plan(
    workflow: Workflow,
    config: ModelConfig,
    *,
    policy_ids: list[str] | None = None,
    case_ids: list[str] | None = None,
    limit: int | None = None,
) -> Plan:
    known = [p.policy_id for p in workflow.bundle.policies]
    if policy_ids is not None and (not policy_ids or set(policy_ids) - set(known)):
        raise ValueError(f"Choose policy IDs from {known}")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    all_cases = {c.case_id for c in workflow.bundle.cases}
    selected = sorted(set(case_ids) if case_ids is not None else all_cases)
    if set(selected) - all_cases:
        raise ValueError(f"Unknown case IDs: {sorted(set(selected) - all_cases)}")
    return Plan(
        model=config,
        policy_ids=[p for p in known if policy_ids is None or p in policy_ids],
        case_ids=selected[:limit],
    )


def run_configuration(workflow: Workflow, plan: Plan, prices: dict) -> dict:
    """Inputs whose fingerprint determines whether a run can resume."""
    packages = ("typesafe-sdk", "system-one-adapter", "pydantic-ai-slim", "pydantic")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    return {
        "model": plan.model.model_dump(mode="json"),
        "policy_ids": plan.policy_ids,
        "prices": prices,
        "dataset_revision": workflow.bundle.dataset_revision,
        "dataset_source": workflow.bundle.dataset_source,
        "policies": [p.model_dump(mode="json") for p in workflow.bundle.policies],
        "infra_revision": revision(Path(__file__).parent),
        "workflow_revision": workflow.implementation_revision,
        "dependencies": versions,
        "dependency_lock": fingerprint(lock_path.read_text()) if lock_path.exists() else None,
        "structured_outputs": True,
        "answer_mode": "probabilities",
        "normalize_probabilities": True,
    }


def default_run_id(plan: Plan, digest: str) -> str:
    model = plan.model
    return f"{model.provider}-{model.model.rsplit('/', 1)[-1]}-{model.thinking}-{digest[:12]}"


@contextmanager
def owned(output: Path, manifest: dict) -> Iterator[None]:
    """Lock the output directory and verify its configuration before resuming."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another runner is using {output}") from exc
        path = output / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError(
                    "Output belongs to a different model, effort, dataset, policy or runtime. "
                    "Choose a different --name."
                )
        elif any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("Output is nonempty but has no manifest; choose a different --name")
        write_text(path, json.dumps(manifest, indent=2) + "\n")
        yield


def load_checkpoints(output: Path, hashes: dict[str, str]) -> dict[str, CaseResult]:
    results: dict[str, CaseResult] = {}
    for path in sorted((output / "cases").glob("*.json")):
        result = CaseResult.model_validate_json(path.read_text())
        if result.case_hash != hashes.get(result.case_id):
            raise ValueError(f"Checkpoint does not match the dataset: {path}")
        if result.case_id in results:
            raise ValueError(f"Duplicate case checkpoint: {result.case_id}")
        results[result.case_id] = result
    return results


def run_case(workflow: Workflow, plan: Plan, case_id: str, case_hash: str, session: Session) -> CaseResult:
    start, error = time.perf_counter(), None
    try:
        result = workflow.execute(case_id, session)
    except Exception as exc:
        error = describe(exc)
        result = CaseResult(
            case_id=case_id,
            decisions=[
                Decision.failed(p.policy_id, p.revision, error)
                for p in workflow.bundle.policies
                if p.policy_id in plan.policy_ids
            ],
        )
    result.case_hash, result.input_status = case_hash, "verified"
    failed = [d for d in result.decisions if d.status == "error"]
    if failed and error is None:
        error = "; ".join(f"{d.policy_id}: {d.reason}" for d in failed)
    result.execution = session.execution(plan.policy_ids, time.perf_counter() - start, error)
    if failed and any(d.status == "ok" for d in result.decisions):
        result.execution.status = "partial"
    result.calls, result.observations = session.calls, session.observations
    return CaseResult.model_validate(result.model_dump(mode="json"))


def checkpoint(output: Path, result: CaseResult, *, archive_previous: bool) -> None:
    path = output / "cases" / f"{fingerprint(result.case_id)}.json"
    if archive_previous and path.exists():
        previous = CaseResult.model_validate_json(path.read_text())
        write_record(output / "attempts" / f"{path.stem}-{time.time_ns()}.json", previous)
    write_record(path, result)


def save_bundle(output: Path, workflow: Workflow, run: Run, results: dict[str, CaseResult]) -> Results:
    bundle = workflow.bundle.model_copy(deep=True)
    bundle.runs = [run.model_copy(update={"cases": [results[cid] for cid in sorted(results)]})]
    bundle = Results.model_validate(bundle.model_dump(mode="json"))
    write_record(output / "results.json", bundle)
    return bundle


def execute(
    workflow: Workflow,
    plan: Plan,
    output: Path,
    *,
    prices: dict,
    workers: int = 4,
    question_workers: int = 8,
    run_id: str | None = None,
    retry_errors: bool = False,
    client_factory: Callable[[ModelConfig], QuestionClient] = make_client,
    model_factory: Callable = resolve_model,
    progress: Callable[[str], None] = print,
) -> Results:
    if workers < 1 or question_workers < 1:
        raise ValueError("Workers must be positive")
    configuration = run_configuration(workflow, plan, prices)
    digest = fingerprint(configuration)
    run = Run(
        run_id=run_id or default_run_id(plan, digest),
        model=plan.model.identity,
        configuration=configuration,
        configuration_hash=digest,
    )
    hashes = {c.case_id: c.content_hash for c in workflow.bundle.cases}
    manifest = {"run_id": run.run_id, "configuration_hash": digest, "configuration": configuration}
    with owned(output, manifest):
        results = load_checkpoints(output, hashes)
        pending = [
            cid
            for cid in plan.case_ids
            if cid not in results
            or (retry_errors and results[cid].execution and results[cid].execution.status != "ok")
        ]

        def one(cid: str) -> CaseResult:
            # Isolate provider state and diagnostics per active case.
            client = client_factory(plan.model)
            try:
                session = Session(
                    client,
                    model_factory(plan.model),
                    plan.model,
                    prices,
                    policy_ids=plan.policy_ids,
                    question_workers=question_workers,
                )
                result = run_case(workflow, plan, cid, hashes[cid], session)
                checkpoint(output, result, archive_previous=retry_errors)
                return result
            finally:
                close = getattr(client, "close", None)
                if close:
                    close()

        progress(
            f"{workflow.bundle.eval_id}: {len(plan.case_ids)} selected; {len(pending)} pending; "
            f"{len(plan.case_ids) - len(pending)} checkpointed"
        )
        save_bundle(output, workflow, run, results)
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(one, cid): cid for cid in pending}
            for future in as_completed(futures):
                result = future.result()
                results[result.case_id] = result
                execution = result.execution
                elapsed = f"{execution.wall_time_s:.2f}s" if execution.wall_time_s is not None else "unknown"
                cost = f"${execution.cost.usd:.6f}" if execution.cost is not None else "unknown"
                progress(
                    f"{result.case_id}: {execution.status} | time={elapsed} | "
                    f"estimated cost={cost} ({len(results)} saved)"
                )
        finally:
            # Finish active cases after interruption; cancel queued work.
            pool.shutdown(wait=True, cancel_futures=True)
            # Include completed checkpoints even if interruption prevented collecting their futures.
            bundle = save_bundle(output, workflow, run, load_checkpoints(output, hashes))
        return bundle
