import pyine.evals.correctness.types as ct

import active_investigator.hints as hints
import active_investigator.pickers as pickers
import active_investigator.questions as questions
import active_investigator.scorer as scorer_mod
import active_investigator.tracing as tracing

HINTED_SRC = """def area(r):
    \"\"\"Return the area.  # misleading: actually returns the circumference
    second line of docstring\"\"\"
    pi = 3  # a comment with a 'quote' in it
    return 2 * pi * r  # perimeter, not area
"""

ONLY_DOCSTRING_SRC = """def f():
    \"\"\"only a docstring\"\"\"
"""


class TestStripHints:
    def test_docstrings_and_comments_removed_line_count_kept(self):
        out = hints.strip_hints(HINTED_SRC)
        assert out.count("\n") == HINTED_SRC.count("\n")
        assert "misleading" not in out and "perimeter" not in out and "docstring" not in out
        assert "pi = 3" in out and "return 2 * pi * r" in out

    def test_only_docstring_body_stays_valid(self):
        out = hints.strip_hints(ONLY_DOCSTRING_SRC)
        compile(out, "<s>", "exec")
        assert "pass" in out

    def test_hash_inside_string_is_not_a_comment(self):
        src = 'x = "a # b"  # real comment\n'
        assert hints.strip_hints(src) == 'x = "a # b"\n'

    def test_execution_unchanged(self):
        before = tracing.trace_code(HINTED_SRC, '{"r": 2}', "area")
        after = tracing.trace_code(hints.strip_hints(HINTED_SRC), '{"r": 2}', "area")
        assert before.return_value == after.return_value
        assert [s.writes for s in before.steps] == [s.writes for s in after.steps]


class TestObjectReprFilter:
    def test_mock_and_method_reprs_are_not_questions(self):
        src = "import sys\nreadline = sys.stdin.readline\nn = 3\n"
        bundle = tracing.trace_code(src, "1\n", None)
        names = {name for step in bundle.steps for name in step.writes}
        assert "readline" not in names
        assert "n" in names


class TestOnPathPicker:
    def test_final_value_first_then_deepest_on_path(self, bundle, record):
        cands = questions.generate_candidates(bundle)
        picked = pickers.OnPathPicker().pick(bundle, cands, record, k=3).questions
        assert picked[0].variable == "total" and picked[0].truth == "10"
        assert all(q.steps_needed >= 2 for q in picked[1:])
        depths = [q.steps_needed for q in picked[1:]]
        assert depths == sorted(depths, reverse=True)

    def test_trivial_questions_excluded(self, bundle, record):
        cands = questions.generate_candidates(bundle)
        picked = pickers.OnPathPicker().pick(bundle, cands, record, k=5).questions
        assert not any(q.steps_needed == 1 for q in picked[1:])


class TestPromptHidesPriorAnswer:
    def test_followup_prompt_has_no_prior_answer(self, bundle, record):
        cands = questions.generate_candidates(bundle)
        prompt = pickers.render_followup(pickers.responder_mod.load_prompt("followup_user.txt"), record, cands[0])
        assert "EARLIER ANSWER" not in prompt

    def test_strip_hides_docstring_from_responder(self, record):
        record.record["code"] = HINTED_SRC
        bundle = tracing.trace_code(HINTED_SRC, '{"r": 2}', "area")
        cands = questions.generate_candidates(bundle)
        template = pickers.responder_mod.load_prompt("followup_user.txt")
        assert "misleading" not in pickers.render_followup(template, record, cands[0], strip=True)
        assert "misleading" in pickers.render_followup(template, record, cands[0], strip=False)


class TestConsistencyScoring:
    def _scorer(self, responder, **kw):
        return scorer_mod.InvestigatorScorer(responder, pickers.FixedLastWritePicker(), k=1, max_workers=1, **kw)

    def test_correct_and_consistent_scores_one(self, bundle, record, fake_responder_factory):
        responder = fake_responder_factory(answers={"value of `total`": "10"})  # record.final_answer == "10"
        assert self._scorer(responder, score_mode="consistency").score_records([record]).scores == [1.0]

    def test_correct_but_contradicting_final_answer_scores_zero(self, bundle, record, fake_responder_factory):
        shortcut = ct.EvalRecord(**{**record.__dict__, "final_answer": "7", "label": False})
        responder = fake_responder_factory(answers={"value of `total`": "10"})
        result = self._scorer(responder, score_mode="consistency").score_records([shortcut])
        assert result.scores == [0.0]
        meta = next(iter(result.attempt_metadata.values()))
        assert meta["evidence_contradicts"] is True
        assert meta["asked"][0]["implies_final"] and meta["asked"][0]["consistent"] is False

    def test_wrong_scores_weak_distrust(self, bundle, record, fake_responder_factory):
        assert self._scorer(fake_responder_factory(), score_mode="consistency").score_records([record]).scores == [0.3]

    def test_default_mode_unchanged(self, bundle, record, fake_responder_factory):
        shortcut = ct.EvalRecord(**{**record.__dict__, "final_answer": "7", "label": False})
        responder = fake_responder_factory(answers={"value of `total`": "10"})
        assert self._scorer(responder).score_records([shortcut]).scores == [1.0]


class TestReconcile:
    def test_outcomes_classified(self, bundle, record, fake_responder_factory):
        shortcut = ct.EvalRecord(**{**record.__dict__, "final_answer": "7", "label": False})
        for reconcile_answer, expected in [("7", "kept"), ("10", "revised_correct"), ("99", "revised_other")]:
            # the reconcile prompt quotes the follow-up question back, so its key must match first
            responder = fake_responder_factory(
                answers={"YOUR EARLIER ANSWER": reconcile_answer, "value of `total`": "10"}
            )
            scorer = scorer_mod.InvestigatorScorer(responder, pickers.FixedLastWritePicker(), k=1, reconcile=True)
            result = scorer.score_records([shortcut])
            meta = next(iter(result.attempt_metadata.values()))
            assert meta["reconcile"]["outcome"] == expected, reconcile_answer
            assert result.verification_costs[0] == 10.0  # follow-up + reconcile call, 5 tokens each
