from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from core.answers import expected, p_at_least
from core.question_types import Answers, Choice, Score
from .questions import spec_of


class Cond:
    def holds(self, ans: Answers, case) -> bool: raise NotImplementedError
    def describe(self) -> str: raise NotImplementedError
    def __and__(self, other: "Cond") -> "Cond":
        left = list(self.conds) if isinstance(self, All) else [self]
        right = list(other.conds) if isinstance(other, All) else [other]
        return All(left + right)
    def __or__(self, other: "Cond") -> "Cond":
        left = list(self.conds) if isinstance(self, Any_) else [self]
        right = list(other.conds) if isinstance(other, Any_) else [other]
        return Any_(left + right)
    def __invert__(self) -> "Cond": return Not(self)


@dataclass(frozen=True)
class ChoiceTop(Cond):
    q: str
    threshold: float
    option: Optional[str] = None

    def holds(self, ans, case):
        a = ans.get(self.q)
        return a is not None and a.confidence > self.threshold and (self.option is None or a.choice == self.option)

    def describe(self):
        if self.option:
            return f"`{self.q}` is `{self.option}` with confidence > {self.threshold}"
        return f"`{self.q}` confidence > {self.threshold}"


@dataclass(frozen=True)
class ChoiceIn(Cond):
    q: str
    options: Sequence[str]

    def __post_init__(self):
        spec = spec_of(self.q)
        assert isinstance(spec, Choice), self.q
        if spec.criteria:
            bad = set(self.options) - set(spec.criteria)
            assert not bad, bad

    def holds(self, ans, case):
        return self.q in ans and ans[self.q].choice in self.options

    def describe(self):
        return f"`{self.q}` is one of {{{', '.join(f'`{o}`' for o in self.options)}}}"


@dataclass(frozen=True)
class ScoreTail(Cond):
    q: str
    level: int
    threshold: float
    op: str = ">"

    def __post_init__(self):
        spec = spec_of(self.q)
        assert isinstance(spec, Score), self.q
        assert 1 <= self.level <= len(spec.criteria) - 1, (self.q, self.level)
        assert self.op in (">", "<")

    def holds(self, ans, case):
        if self.q not in ans:
            return False
        m = p_at_least(ans[self.q], self.level)
        return m > self.threshold if self.op == ">" else m < self.threshold

    def describe(self):
        return f"P(`{self.q}` ≥ {self.level}) {self.op} {self.threshold}"


@dataclass(frozen=True)
class ScoreExpect(Cond):
    q: str
    threshold: float
    threshold2: float

    def holds(self, ans, case):
        a = ans.get(self.q)
        return a is not None and expected(a) > self.threshold and a.confidence > self.threshold2

    def describe(self):
        return f"E[`{self.q}`] > {self.threshold} and `{self.q}` confidence > {self.threshold2}"


@dataclass(frozen=True)
class NoulGate(Cond):
    q: str
    threshold: float = 0.5

    def __post_init__(self):
        assert self.threshold >= 0.5, (self.q, self.threshold)

    def holds(self, ans, case):
        return self.q in ans and ans[self.q].noul > self.threshold

    def describe(self):
        return f"`{self.q}` > {self.threshold}"


@dataclass(frozen=True)
class Fact(Cond):
    text: str
    fn: Callable = field(compare=False, hash=False)

    def holds(self, ans, case): return bool(self.fn(case))
    def describe(self): return self.text


@dataclass(frozen=True)
class All(Cond):
    conds: Sequence[Cond]
    def holds(self, ans, case): return all(c.holds(ans, case) for c in self.conds)
    def describe(self): return " and ".join(_paren(c) for c in self.conds)


@dataclass(frozen=True)
class Any_(Cond):
    conds: Sequence[Cond]
    def holds(self, ans, case): return any(c.holds(ans, case) for c in self.conds)
    def describe(self): return " or ".join(_paren(c) for c in self.conds)


@dataclass(frozen=True)
class Not(Cond):
    cond: Cond
    def holds(self, ans, case): return not self.cond.holds(ans, case)
    def describe(self): return f"not ({self.cond.describe()})"


class Always(Cond):
    def holds(self, ans, case): return True
    def describe(self): return "always (default)"


@dataclass(frozen=True)
class NodeRan(Cond):
    node: str
    def holds(self, ans, case): return f"__ran__{self.node}" in ans
    def describe(self): return f"the `{self.node}` node ran"


def _paren(c: Cond) -> str:
    d = c.describe()
    return f"({d})" if isinstance(c, (All, Any_)) else d


def forms_used(cond: Cond) -> set:
    if isinstance(cond, (All, Any_)):
        return set().union(*(forms_used(c) for c in cond.conds))
    if isinstance(cond, Not):
        return forms_used(cond.cond)
    if isinstance(cond, (ChoiceTop, ChoiceIn, ScoreTail, ScoreExpect, NoulGate)):
        return {type(cond).__name__}
    return set()
