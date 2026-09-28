from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as parquet
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

from .contract import Action, CaseResult, Decision, LabelSet, ModelIdentity, Results, fingerprint

DATASETS = {
    "invoice_processing": "typesafe/evalsafe-invoice-processing",
    "customer_service": "typesafe/evalsafe-customer-service",
    "security_incidents": "typesafe/evalsafe-security-incidents",
    "agent_trace_observability": "typesafe/evalsafe-agent-trace-observability",
}


def load_dataset(workflow: str, revision: str | None = None) -> Dataset:
    repo_id = DATASETS[workflow]
    try:
        root = Path(
            hf_hub_download(repo_id, "dataset.json", repo_type="dataset", revision=revision or "main")
        )
        root = root.parent
        hf_hub_download(repo_id, "data/cases.parquet", repo_type="dataset", revision=root.name)
    except RepositoryNotFoundError as exc:
        raise ValueError(
            f"Cannot access {repo_id}. For private datasets, run `uv run hf auth login` or set HF_TOKEN."
        ) from exc
    dataset = Dataset(repo_id, root)
    if dataset.manifest["workflow"] != workflow:
        raise ValueError(f"{repo_id} does not contain the {workflow} workflow")
    return dataset


class Dataset:
    def __init__(self, repo_id: str, root: Path):
        self.source = f"https://huggingface.co/datasets/{repo_id}/tree/{root.name}"
        self.manifest = json.loads((root / "dataset.json").read_text())
        self.rows = parquet.read_table(root / "data/cases.parquet").to_pylist()
        self.inputs = {row["case_id"]: json.loads(row["input_json"]) for row in self.rows}
        self.metadata = {row["case_id"]: json.loads(row["metadata_json"]) for row in self.rows}
        if not self.rows or len(self.inputs) != len(self.rows):
            raise ValueError("Dataset must contain nonempty, unique cases")

    def attach_labels(self, bundle: Results) -> None:
        policies = {p.policy_id: p.revision for p in bundle.policies}
        if set(policies) != {p["policy_id"] for p in self.manifest["policies"]}:
            raise ValueError("Dataset policies do not match the workflow")
        hashes = {c.case_id: c.content_hash for c in bundle.cases}
        for role in self.manifest["labelsets"]:
            sources, cases = {}, []
            for row in self.rows:
                label = row[role]
                contributors = label["contributors"] if role == "consensus" else [role]
                for contributor in contributors:
                    model = row[contributor]["model"]
                    if model is not None:
                        identity = ModelIdentity(
                            name=f"{model['provider']}:{model['name']}",
                            lab=model["provider"],
                            effort=model["reasoning_effort"],
                        )
                        sources[identity.model_dump_json()] = identity
                decisions = [
                    Decision(
                        policy_id=d["policy_id"],
                        policy_revision=policies[d["policy_id"]],
                        status=d["status"],
                        reason=d.get("reason"),
                        actions=[action(a) for a in d["actions"]],
                        primary_action=action(d["primary_action"]) if d["primary_action"] else None,
                    )
                    for d in label["decisions"]
                ]
                if not decisions:
                    decisions = [
                        Decision(
                            policy_id=policy,
                            policy_revision=revision,
                            status="missing",
                            reason=label.get("reason"),
                        )
                        for policy, revision in policies.items()
                    ]
                cases.append(
                    CaseResult(
                        case_id=row["case_id"],
                        case_hash=hashes[row["case_id"]],
                        input_status="bound_annotation",
                        decisions=decisions,
                        source=f"{self.source}#{row['case_id']}",
                        details={k: v for k, v in label.items() if k != "decisions"},
                    )
                )
            label_set = LabelSet(
                label_set_id=f"labels-{role}",
                title=f"{role} reference",
                revision=fingerprint([{r["case_id"]: r[role]} for r in self.rows]),
                method="consensus" if role == "consensus" else "independent",
                sources=[sources[k] for k in sorted(sources)],
                cases=cases,
            )
            bundle.label_sets.append(label_set)
            bundle.references[role] = label_set.label_set_id
        bundle.dataset_source = self.source


def action(value: dict) -> Action:
    return Action(name=value["name"], arguments=json.loads(value["arguments_json"]))
