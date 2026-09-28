from collections.abc import Iterator, Mapping
from typing import Any

from .question_types import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
)


def ranked(answer: ChoiceAnswer) -> list[tuple[str, float]]:
    return sorted(answer.probabilities.items(), key=lambda kv: -kv[1])


def gap(answer: ChoiceAnswer) -> float:
    order = ranked(answer)
    return order[0][1] - order[1][1] if len(order) > 1 else 1.0


def expected(answer: ScoreAnswer) -> float:
    return sum(int(k) * v for k, v in answer.probabilities.items())


def p_at_least(answer: ScoreAnswer, level: int) -> float:
    return sum(v for k, v in answer.probabilities.items() if int(k) >= level)


class NodeAnswers(Mapping[str, Answer]):
    """SDK answers accessible by question ID or attribute."""

    def __init__(self, answers: Mapping[str, Answer]) -> None:
        self._answers = dict(answers)

    def __getitem__(self, key: str) -> Answer:
        return self._answers[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._answers)

    def __len__(self) -> int:
        return len(self._answers)

    def __getattr__(self, name: str) -> Answer:
        try:
            return self.__dict__["_answers"][name]
        except KeyError as exc:
            raise AttributeError(f"no answer {name!r} in this node") from exc

    def to_json(self) -> dict[str, Any]:
        out = {}
        for key, answer in self.items():
            if isinstance(answer, NoulAnswer):
                out[key] = {"type": "noul", "p": answer.noul}
            else:
                field = "choice" if isinstance(answer, ChoiceAnswer) else "value"
                out[key] = {
                    "type": answer.type,
                    field: answer.choice if isinstance(answer, ChoiceAnswer) else answer.score,
                    "confidence": answer.confidence,
                    "probabilities": {str(k): v for k, v in answer.probabilities.items()},
                }
        return out


def from_response(response: SystemOneResponse) -> NodeAnswers:
    """Preserve the returned choice and score; treat missing confidence as zero."""
    return NodeAnswers(
        {
            key: answer
            if isinstance(answer, NoulAnswer)
            else answer.model_copy(update={"confidence": float(answer.confidence or 0.0)})
            for key, answer in response.answers.items()
        }
    )


def from_json(blob: Mapping[str, Any]) -> NodeAnswers:
    answers = {}
    for key, value in blob.items():
        kind = value["type"]
        if kind == "noul":
            answers[key] = NoulAnswer(noul=float(value["p"]))
        elif kind == "choice":
            answers[key] = ChoiceAnswer(
                choice=value["choice"],
                confidence=float(value["confidence"]),
                probabilities={k: float(v) for k, v in value["probabilities"].items()},
            )
        elif kind == "score":
            answers[key] = ScoreAnswer(
                score=float(value["value"]),
                confidence=float(value["confidence"]),
                legend={},
                probabilities={int(k): float(v) for k, v in value["probabilities"].items()},
            )
        else:
            raise ValueError(f"unknown answer kind {kind!r} for {key!r}")
    return NodeAnswers(answers)
