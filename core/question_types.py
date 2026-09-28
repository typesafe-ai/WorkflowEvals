from typesafe_sdk import (
    Answer as Answer,
    Choice as Choice,
    ChoiceAnswer as ChoiceAnswer,
    Noul as Noul,
    NoulAnswer as NoulAnswer,
    NoulCriteria as NoulCriteria,
    Score as Score,
    ScoreAnswer as ScoreAnswer,
    SystemOneResponse as SystemOneResponse,
)

type Question = Noul | Choice | Score
type QuestionSet = dict[str, Question]
type Answers = dict[str, Answer]


def question_record(question: Question) -> dict:
    """Keep explicit optional nulls so saved question fingerprints remain stable."""
    raw = question.model_dump(mode="json")
    criteria = raw.get("criteria")
    if isinstance(question, Noul) and criteria is not None:
        criteria = {"true": criteria.get("true"), "false": criteria.get("false")}
    return {"type": question.type, "instructions": raw.get("instructions"), "criteria": criteria}


def response_record(response: SystemOneResponse) -> dict:
    """Omit diagnostics and rubric text already recorded in the questions."""
    raw = response.model_dump(mode="json", exclude={"debug"})
    for answer in raw["answers"].values():
        answer.pop("legend", None)
    return raw
