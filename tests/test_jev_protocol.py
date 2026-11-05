import math
import unittest

from docs_portable import jev_protocol as proto


def words(text):
    """Fake tokenizer: one id per whitespace-separated word (newlines count as their own token)."""
    return [hash(w) % 1000 for w in text.replace("\n", " \n ").split(" ") if w]


class FakeRun:
    """Prefers options containing the word "good"; records every decision it gets."""

    def __init__(self):
        self.calls = []

    def __call__(self, decisions):
        self.calls.append(list(decisions))
        out = []
        for d in decisions:
            weights = [5.0 if "good" in o else 1.0 for o in d.options]
            out.append([w / sum(weights) for w in weights])
        return out


class PlanTest(unittest.TestCase):
    def test_noul_criteria_are_appended(self):
        p = proto.plan("q", {"type": "noul", "instructions": "Refund?", "criteria": {"true": "asks", "false": "no"}})
        self.assertEqual(p.lines, ["false", "true"])
        self.assertEqual(p.question, "Refund? Criteria: true = asks; false = no.")

    def test_choice_lines_and_structured_values(self):
        p = proto.plan("q", {"type": "choice", "instructions": {"ask": "which"},
                             "criteria": {"a": None, "b": "line one\nline two", "c": {"k": 1}}})
        self.assertEqual(p.question, '{"ask": "which"}')
        self.assertEqual(p.lines, ["a", "b: line one line two", 'c: {"k": 1}'])
        self.assertEqual(p.keys, ["a", "b", "c"])

    def test_score_becomes_choice_over_levels(self):
        p = proto.plan("q", {"type": "score", "instructions": "How bad?", "criteria": ["fine", "bad", "awful"]})
        self.assertEqual((p.type, p.keys, p.lines), ("score", ["0", "1", "2"], ["fine", "bad", "awful"]))
        self.assertEqual(p.legend, {"0": "fine", "1": "bad", "2": "awful"})

    def test_rejects_malformed_questions(self):
        bad = [
            {"type": "maybe", "instructions": "x"},
            {"type": "noul"},
            {"type": "noul", "instructions": "  "},
            {"type": "noul", "instructions": "x", "criteria": ["yes"]},
            {"type": "noul", "instructions": "x", "criteria": {"yes": "y"}},
            {"type": "choice", "instructions": "x", "criteria": {"only": None}},
            {"type": "choice", "instructions": "x", "criteria": ["a", "b"]},
            {"type": "choice", "instructions": "x", "criteria": {str(i): None for i in range(proto.MAX_OPTIONS + 1)}},
            {"type": "score", "instructions": "x", "criteria": ["one"]},
            {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]},
            "not an object",
        ]
        for raw in bad:
            with self.subTest(raw=str(raw)[:60]), self.assertRaises(proto.RequestError):
                proto.plan("q", raw)

    def test_request_limits(self):
        good = {"type": "noul", "instructions": "x"}
        for questions in (None, {}, [good], {f"q{i}": good for i in range(proto.MAX_QUESTIONS + 1)},
                          {"": good}, {"x" * 201: good}):
            with self.subTest(n=str(questions)[:40]), self.assertRaises(proto.RequestError):
                proto.plan_request(questions)

    def test_unserializable_json_is_a_client_error(self):
        deep: list = []
        for _ in range(100_000):
            deep = [deep]
        with self.assertRaises(proto.RequestError):
            proto.render(deep)


class StateTest(unittest.TestCase):
    def test_renders_strings_and_json(self):
        self.assertEqual(proto.check_state("hi"), "hi")
        self.assertEqual(proto.check_state({"a": "ü"}), '{"a": "ü"}')
        self.assertEqual(proto.check_state(None), "")

    def test_rejects_images(self):
        for state in (["look", {"image": "data:image/png;base64,AA"}], [{"type": "image_url", "image_url": {}}]):
            with self.subTest(state=state), self.assertRaisesRegex(proto.RequestError, "images"):
                proto.check_state(state)
        # A JSON record that merely has an "image" field is ordinary state.
        self.assertIn("image", proto.check_state([{"image": "cat.png", "caption": "a cat"}]))


class EvaluateTest(unittest.TestCase):
    def test_small_questions_take_one_batched_round(self):
        plans = proto.plan_request({
            "n": {"type": "noul", "instructions": "x"},
            "c": {"type": "choice", "instructions": "x", "criteria": {"bad": None, "good": None}},
            "s": {"type": "score", "instructions": "x", "criteria": ["bad", "good", "meh"]},
        })
        run = FakeRun()
        probs = proto.evaluate(plans, run)
        self.assertEqual(len(run.calls), 1)
        self.assertEqual([d.kind for d in run.calls[0]], ["noul", "choice", "choice"])
        self.assertAlmostEqual(probs[1][1], 5 / 6)
        answers = [proto.answer(p, x) for p, x in zip(plans, probs)]
        self.assertEqual(answers[1]["choice"], "good")
        self.assertEqual(answers[2]["type"], "score")
        self.assertAlmostEqual(answers[2]["score"], (5 + 2) / 7)

    def test_tournament_covers_all_options_and_sums_to_one(self):
        criteria = {f"opt{i}": ("good" if i in (3, 40, 99) else "meh") for i in range(100)}
        plans = proto.plan_request({"c": {"type": "choice", "instructions": "x", "criteria": criteria}})
        run = FakeRun()
        (probs,) = proto.evaluate(plans, run)
        self.assertEqual(len(run.calls), 2)
        groups = run.calls[0]
        self.assertEqual(len(groups), 7)  # ceil(100 / 16), balanced
        self.assertTrue(all(14 <= len(d.options) <= 15 for d in groups))
        self.assertLessEqual(len(run.calls[1][0].options), proto.HEAD_OPTIONS)
        self.assertEqual(len(probs), 100)
        self.assertAlmostEqual(sum(probs), 1.0)
        best = sorted(range(100), key=lambda i: -probs[i])[:3]
        self.assertEqual(sorted(best), [3, 40, 99])
        self.assertIn(proto.answer(plans[0], probs)["choice"], {"opt3", "opt40", "opt99"})

    def test_maximum_options_still_fit_one_final(self):
        criteria = {f"o{i}": None for i in range(proto.MAX_OPTIONS)}
        plans = proto.plan_request({"c": {"type": "choice", "instructions": "x", "criteria": criteria}})
        run = FakeRun()
        (probs,) = proto.evaluate(plans, run)
        self.assertEqual(len(run.calls[1][0].options), proto.HEAD_OPTIONS)
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_invalid_model_output_is_rejected(self):
        plans = proto.plan_request({"n": {"type": "noul", "instructions": "x"}})
        for bad in ([[0.5]], [[math.nan, 0.5]], [[-0.1, 1.1]], []):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                proto.evaluate(plans, lambda decisions, out=bad: out)


class ConfidenceTest(unittest.TestCase):
    def test_choice_formula(self):
        self.assertAlmostEqual(proto.choice_confidence([0.6, 0.3, 0.1]), 0.4)
        self.assertAlmostEqual(proto.choice_confidence([1 / 3] * 3), 0.0)
        self.assertAlmostEqual(proto.choice_confidence([1.0, 0.0]), 1.0)

    def test_score_formula_matches_typesafe_examples(self):
        self.assertAlmostEqual(proto.score_confidence([0, 0.5, 0.5]), 0.25)
        self.assertAlmostEqual(proto.score_confidence([0.5, 0, 0.5]), 0.0)
        self.assertAlmostEqual(proto.score_confidence([0, 0, 1, 0, 0]), 1.0)


class PromptTest(unittest.TestCase):
    def test_layout(self):
        ids = proto.prompt_ids(words, proto.Decision("noul", "q?", ("false", "true")), words("the state"), 100, 1000)
        expected = words("[kind] noul\n[state] ") + words("the state") + words("\n[question] q?\n[options]\n") \
            + words("false") + words("\n") + words("true") + words("\n[decision]:")
        self.assertEqual(ids, expected)

    def test_choice_options_get_letters(self):
        ids = proto.prompt_ids(words, proto.Decision("choice", "q", ("x", "y")), [], 100, 1000)
        tail = words("A) x") + words("\n") + words("B) y") + words("\n[decision]:")
        self.assertEqual(ids[-len(tail):], tail)

    def test_long_state_keeps_head_and_tail(self):
        state = list(range(10_000, 12_000))
        ids = proto.prompt_ids(words, proto.Decision("noul", "q", ("false", "true")), state, 100, 1000)
        kept = [t for t in ids if t >= 10_000]
        self.assertEqual(kept, state[:60] + state[-40:])

    def test_prompt_never_exceeds_the_cap(self):
        options = tuple(" ".join(["word"] * 400) for _ in range(16))
        state = list(range(10_000, 15_000))
        ids = proto.prompt_ids(words, proto.Decision("choice", "q", options), state, 1024, 2048)
        self.assertLessEqual(len(ids), 2048)
        self.assertGreaterEqual(len([t for t in ids if t >= 10_000]), 64)

    def test_question_too_long_is_a_client_error(self):
        with self.assertRaises(proto.RequestError):
            proto.prompt_ids(words, proto.Decision("noul", "w " * 5000, ("false", "true")), [], 100, 1000)


if __name__ == "__main__":
    unittest.main()
