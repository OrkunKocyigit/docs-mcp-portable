"""TypeSafe System One requests ({model, state, questions}) translated to JEV-9B System 1 decisions.

JEV-9B answers three fixed shapes in one forward pass each: noul (false/true), score (0-5, no level text) and
choice (2-16 plain options). System One questions are wider, so this module maps them:

* noul: criteria (what true / false mean) are appended to the question.
* choice: option lines are "key" or "key: description"; more than 16 options run as a tournament (balanced groups
  of <= 16, then a final over the best of each group), as in the model's own reference server.
* score: levels are described in text, which the 0-5 head never saw, so a score is asked as a choice over the
  level descriptions and the score is the probability-weighted level.

No torch here; the model sits behind ``Run``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

TOKEN_ENV = "DOCS_PORTABLE_JEV_TOKEN"  # per-run bearer token shared by the launcher, server and jev-mcp
HEAD_OPTIONS = 16  # choice slots of the decision head (A-P)
MAX_OPTIONS = HEAD_OPTIONS * HEAD_OPTIONS  # one tournament round: 16 groups, one finalist each
MAX_SCORE_LEVELS = 10
MAX_QUESTIONS = 256
MAX_QUESTION_ID_CHARS = 200
NOUL_OPTIONS = ("false", "true")
LABELS = "ABCDEFGHIJKLMNOP"


class RequestError(ValueError):
    """A client error; the message is safe to return."""


@dataclass(frozen=True)
class Decision:
    """One System 1 pass: the head kind ("noul" or "choice"), question text and option lines."""

    kind: str
    question: str
    options: tuple[str, ...]


# Runs decisions against one state and returns a probability list per decision, aligned with its options.
Run = Callable[[Sequence[Decision]], list[list[float]]]


def render(value: Any) -> str:
    """Instructions, criteria and state may be strings or JSON; JSON is passed as compact text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise RequestError("instructions, criteria and state must be strings or JSON values") from exc


def _line(text: str) -> str:
    # Options are newline-separated in the prompt.
    return " ".join(text.split())


@dataclass
class QuestionPlan:
    qid: str
    type: str  # noul | choice | score (answer shape)
    question: str
    keys: list[str]  # choice: option keys; score: "0".."n-1"
    lines: list[str]  # option lines shown to the model
    legend: dict[str, Any] | None = None

    # Filled during evaluation.
    groups: list[list[int]] | None = None
    finalists: list[int] | None = None


def plan(qid: str, raw: Any) -> QuestionPlan:
    if not isinstance(raw, dict):
        raise RequestError(f"question {qid!r} must be an object")
    kind = raw.get("type")
    instructions = render(raw.get("instructions"))
    criteria = raw.get("criteria")
    if not instructions.strip():
        raise RequestError(f"question {qid!r} needs instructions")

    if kind == "noul":
        question = instructions
        if criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                raise RequestError(f"question {qid!r}: noul criteria must be an object with true and/or false")
            parts = [f"{k} = {_line(render(criteria[k]))}" for k in ("true", "false") if criteria.get(k) is not None]
            if parts:
                question += " Criteria: " + "; ".join(parts) + "."
        return QuestionPlan(qid, "noul", question, list(NOUL_OPTIONS), list(NOUL_OPTIONS))

    if kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
            raise RequestError(f"question {qid!r}: choice criteria must map 2 to {MAX_OPTIONS} options to descriptions")
        keys = [str(k) for k in criteria]
        lines = []
        for key, desc in criteria.items():
            text = _line(render(desc))
            lines.append(f"{_line(key)}: {text}" if text else _line(key))
        return QuestionPlan(qid, "choice", instructions, keys, lines)

    if kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
            raise RequestError(f"question {qid!r}: score criteria must be a list of 2 to {MAX_SCORE_LEVELS} levels")
        lines = [_line(render(level)) or str(i) for i, level in enumerate(criteria)]
        keys = [str(i) for i in range(len(criteria))]
        return QuestionPlan(qid, "score", instructions, keys, lines, legend=dict(zip(keys, criteria)))

    raise RequestError(f"question {qid!r}: type must be noul, choice or score")


def plan_request(questions: Any) -> list[QuestionPlan]:
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise RequestError(f"questions must be an object with 1 to {MAX_QUESTIONS} entries")
    for qid in questions:
        if not qid or len(qid) > MAX_QUESTION_ID_CHARS:
            raise RequestError(f"question ids must be 1 to {MAX_QUESTION_ID_CHARS} characters")
    return [plan(qid, raw) for qid, raw in questions.items()]


def balanced_groups(n: int, size: int = HEAD_OPTIONS) -> list[list[int]]:
    count = math.ceil(n / size)
    base, extra = divmod(n, count)
    groups, start = [], 0
    for g in range(count):
        width = base + (1 if g < extra else 0)
        groups.append(list(range(start, start + width)))
        start += width
    return groups


def _top(indices: list[int], probs: list[float], k: int) -> list[int]:
    ranked = sorted(range(len(indices)), key=lambda i: -probs[i])[:k]
    return [indices[i] for i in sorted(ranked)]  # keep caller order among finalists


def evaluate(plans: list[QuestionPlan], run: Run) -> list[list[float]]:
    """Probabilities per plan, aligned with plan.keys. At most two batched rounds of model passes."""
    first: list[Decision] = []
    owners: list[tuple[int, int]] = []  # (plan index, group index or -1)
    for p_i, p in enumerate(plans):
        kind = "noul" if p.type == "noul" else "choice"
        if len(p.lines) <= HEAD_OPTIONS:
            first.append(Decision(kind, p.question, tuple(p.lines)))
            owners.append((p_i, -1))
            continue
        p.groups = balanced_groups(len(p.lines))
        for g_i, group in enumerate(p.groups):
            first.append(Decision("choice", p.question, tuple(p.lines[i] for i in group)))
            owners.append((p_i, g_i))

    results: list[list[float] | None] = [None] * len(plans)
    group_probs: dict[int, list[list[float]]] = {}
    for (p_i, g_i), probs in zip(owners, _checked(run, first)):
        if g_i < 0:
            results[p_i] = probs
        else:
            group_probs.setdefault(p_i, []).append(probs)

    finals: list[Decision] = []
    final_owner: list[int] = []
    for p_i, probs_per_group in group_probs.items():
        p = plans[p_i]
        per_group = max(1, HEAD_OPTIONS // len(p.groups))
        p.finalists = [i for group, probs in zip(p.groups, probs_per_group) for i in _top(group, probs, per_group)]
        finals.append(Decision("choice", p.question, tuple(p.lines[i] for i in p.finalists)))
        final_owner.append(p_i)

    for p_i, final in zip(final_owner, _checked(run, finals)):
        results[p_i] = _combine(plans[p_i], group_probs[p_i], final)
    return [r for r in results if r is not None]


def _checked(run: Run, decisions: list[Decision]) -> list[list[float]]:
    if not decisions:
        return []
    out = run(decisions)
    if len(out) != len(decisions):
        raise RuntimeError("model returned a wrong number of answers")
    for decision, probs in zip(decisions, out):
        if len(probs) != len(decision.options) or not all(math.isfinite(x) and x >= 0 for x in probs):
            raise RuntimeError("model returned invalid probabilities")
    return out


def _combine(p: QuestionPlan, group_probs: list[list[float]], final: list[float]) -> list[float]:
    """Each group's share comes from the final; inside a group, non-finalists keep their group probabilities.

    For group g with finalists F (final mass w_g) and non-finalist group mass m_g, a finalist gets
    p_final * (1 - m_g) and a non-finalist w_g * p_group, so every group sums to w_g and the total to 1.
    """
    final_of = dict(zip(p.finalists, final))
    out = [0.0] * len(p.lines)
    for group, probs in zip(p.groups, group_probs):
        in_group = dict(zip(group, probs))
        finalists = [i for i in group if i in final_of]
        w = sum(final_of[i] for i in finalists)
        m = sum(q for i, q in in_group.items() if i not in final_of)
        for i in group:
            out[i] = final_of[i] * (1 - m) if i in final_of else w * in_group[i]
    total = sum(out)
    return [x / total for x in out] if total > 0 else [1 / len(out)] * len(out)


def choice_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    return 1.0 if n < 2 else max(0.0, (max(probs) - 1 / n) / (1 - 1 / n))


def score_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    m = max(range(n), key=lambda i: probs[i])
    spread = sum(p * abs(i - m) for i, p in enumerate(probs))
    mad_uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / mad_uniform)


def answer(p: QuestionPlan, probs: list[float]) -> dict[str, Any]:
    if p.type == "noul":
        return {"type": "noul", "noul": probs[1]}
    best = max(range(len(probs)), key=lambda i: probs[i])
    distribution = dict(zip(p.keys, probs))
    if p.type == "choice":
        return {"type": "choice", "choice": p.keys[best], "confidence": choice_confidence(probs),
                "probabilities": distribution}
    return {"type": "score", "score": sum(i * x for i, x in enumerate(probs)), "confidence": score_confidence(probs),
            "legend": p.legend, "probabilities": distribution}


def check_state(state: Any) -> str:
    if isinstance(state, list):
        for part in state:
            if isinstance(part, dict) and ("image" in part and len(part) == 1 or part.get("type") == "image_url"):
                raise RequestError("images are not supported by the local JEV-9B endpoint")
    if state is None:
        return ""
    return render(state)


# ------------------------------------------------------------------ prompts

def prompt_ids(
    tokenize: Callable[[str], list[int]],
    decision: Decision,
    state_ids: list[int],
    max_state_tokens: int,
    max_prompt_tokens: int,
) -> list[int]:
    """Token ids of the System 1 prompt, as JEV-9B was trained:

    [kind] <kind>\\n[state] <state>\\n[question] <question>\\n[options]\\n<options>\\n[decision]:

    The state keeps its first 60% / last 40% of tokens beyond the budget; if the options are too long, each
    option line is cut to an equal share first.
    """
    head = tokenize(f"[kind] {decision.kind}\n[state] ")
    middle = tokenize(f"\n[question] {decision.question}\n[options]\n")
    tail = tokenize("\n[decision]:")
    newline = tokenize("\n")
    if decision.kind == "choice":
        lines = [tokenize(f"{LABELS[i]}) {o}") for i, o in enumerate(decision.options)]
    else:
        lines = [tokenize(o) for o in decision.options]

    fixed = len(head) + len(middle) + len(tail) + len(newline) * (len(lines) - 1)
    min_state = min(len(state_ids), 64)
    room = max_prompt_tokens - fixed - min_state
    if room < 8 * len(lines):
        raise RequestError("a question is too long for [jev] max_prompt_tokens")
    if sum(map(len, lines)) > room:
        cap = room // len(lines)
        lines = [ids[:cap] for ids in lines]
    options: list[int] = []
    for i, ids in enumerate(lines):
        options += (newline if i else []) + ids

    budget = min(max_state_tokens, max_prompt_tokens - fixed - len(options) + len(newline) * (len(lines) - 1))
    budget = max(budget, min_state)
    if len(state_ids) > budget:
        first = budget * 6 // 10
        state_ids = state_ids[:first] + state_ids[len(state_ids) - (budget - first):]
    return head + state_ids + middle + options + tail
