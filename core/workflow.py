from importlib import import_module
from pathlib import Path
from typing import Protocol

from .contract import CaseResult, Results, fingerprint
from .dataset import load_dataset
from .session import Session

class Workflow(Protocol):
    """The bundle contains cases and references; implementation_revision identifies code for resuming."""

    bundle: Results
    implementation_revision: str

    def execute(self, case_id: str, session: Session) -> CaseResult:
        ...


WORKFLOWS = (
    "invoice_processing",
    "customer_service",
    "agent_trace_observability",
    "security_incidents",
)


def load(name: str, dataset_revision: str | None = None) -> Workflow:
    if name not in WORKFLOWS:
        raise ValueError(f"Unknown workflow: {name}")
    root = Path(__file__).resolve().parents[1] / "evals" / name
    dataset = load_dataset(name, dataset_revision)
    workflow = import_module(f"evals.{name}.eval_adapter").Adapter(root, dataset)
    dataset.attach_labels(workflow.bundle)
    Results.model_validate(workflow.bundle.model_dump())
    return workflow


def revision(root: Path) -> str:
    return fingerprint(
        {
            str(p.relative_to(root)): p.read_text()
            for p in sorted(root.rglob("*.py"))
            if not any(s in p.parts for s in (".venv", "tests", "__pycache__"))
        }
    )
