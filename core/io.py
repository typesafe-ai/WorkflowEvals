import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from .contract import CaseResult, Results


def load_results(path: Path) -> Results:
    directory = path if path.is_dir() else None
    bundle = Results.model_validate_json((directory / "results.json" if directory else path).read_text())
    if directory:
        if len(bundle.runs) != 1:
            raise ValueError("A checkpoint directory must contain exactly one run")
        bundle.runs[0].cases = [
            CaseResult.model_validate_json(p.read_text())
            for p in sorted((directory / "cases").glob("*.json"))
        ]
        bundle = Results.model_validate(bundle.model_dump(mode="json"))
    return bundle


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_record(path: Path, record: BaseModel) -> None:
    write_text(path, record.model_dump_json(indent=2) + "\n")
