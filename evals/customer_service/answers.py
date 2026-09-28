from core.question_types import (
    Answer,
    Answers,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    QuestionSet,
    ScoreAnswer,
    SystemOneResponse,
)


def from_system_one_response(response: SystemOneResponse, questions: QuestionSet) -> Answers:
    """Fill every option in question order; derive the choice from its probabilities."""
    out = {}
    for key, question in questions.items():
        answer = response.answers[key]
        if isinstance(question, Noul):
            out[key] = NoulAnswer(noul=float(answer.noul))
        elif isinstance(question, Choice):
            probabilities = {k: float(answer.probabilities.get(k, 0.0)) for k in question.criteria}
            out[key] = ChoiceAnswer(
                choice=max(probabilities, key=probabilities.__getitem__),
                probabilities=probabilities,
                confidence=float(answer.confidence),
            )
        else:
            probabilities = {
                i: float(answer.probabilities.get(i, 0.0)) for i in range(len(question.criteria))
            }
            out[key] = ScoreAnswer(
                score=sum(k * v for k, v in probabilities.items()),
                probabilities=probabilities,
                confidence=float(answer.confidence),
                legend=dict(enumerate(question.criteria)),
            )
    return out


def to_json(answer: Answer) -> dict:
    if isinstance(answer, NoulAnswer):
        return {"type": "noul", "p": round(answer.noul, 4)}
    return {
        "type": answer.type,
        "probs": {str(k): round(v, 4) for k, v in answer.probabilities.items()},
        "confidence": round(float(answer.confidence), 4),
    }


def from_json(data: dict) -> Answer:
    if data["type"] == "noul":
        return NoulAnswer(noul=float(data["p"]))
    confidence = float(data["confidence"])
    if data["type"] == "choice":
        probabilities = {k: float(v) for k, v in data["probs"].items()}
        return ChoiceAnswer(
            choice=max(probabilities, key=probabilities.__getitem__),
            probabilities=probabilities,
            confidence=confidence,
        )
    probabilities = {int(k): float(v) for k, v in data["probs"].items()}
    return ScoreAnswer(
        score=sum(k * v for k, v in probabilities.items()),
        probabilities=probabilities,
        confidence=confidence,
        legend={},
    )
