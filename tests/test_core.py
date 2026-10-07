import active_investigator.grading as grading
import active_investigator.questions as questions
import active_investigator.slicing as slicing


class TestTracing:
    def test_steps_and_return(self, bundle):
        assert bundle.return_value == "10"
        assert bundle.return_step is not None
        assert bundle.steps[0].writes == {"total": "0"}

    def test_last_write_before_return_is_final_total(self, bundle):
        last = bundle.last_write_before_return()
        assert last is not None
        assert last.writes == {"total": "10"}

    def test_occurrence_counts_repeat_lines(self, bundle):
        loop_header = [s for s in bundle.steps if s.line == 3]
        assert [s.occurrence for s in loop_header] == list(range(len(loop_header)))


class TestSlicing:
    def test_accumulator_needs_more_steps_than_index(self, bundle):
        last_total = max(s.step for s in bundle.steps if "total" in s.writes)
        last_n = max(s.step for s in bundle.steps if "n" in s.writes)
        assert len(slicing.steps_needed(bundle, "total", last_total)) > len(slicing.steps_needed(bundle, "n", last_n))

    def test_first_assignment_is_self_contained(self, bundle):
        first = bundle.steps[0]
        assert slicing.steps_needed(bundle, "total", first.step) == [first.step]

    def test_guard_is_included_for_branch_body(self, bundle):
        body_step = next(s for s in bundle.steps if s.line == 5)  # total += n inside the if
        needed = slicing.steps_needed(bundle, "total", body_step.step)
        guard_lines = {s.line for s in bundle.steps if s.step in needed}
        assert 4 in guard_lines


class TestQuestions:
    def test_all_kinds_present_with_truths(self, bundle):
        cands = questions.generate_candidates(bundle)
        kinds = {c.kind for c in cands}
        assert kinds == {"value_at", "branch_at", "execution_count"}
        assert all(c.truth for c in cands)

    def test_no_question_asks_for_the_return_value_directly(self, bundle):
        for cand in questions.generate_candidates(bundle):
            assert "return" not in cand.text.lower().split("`")[0]

    def test_execution_count_truth(self, bundle):
        counts = {c.line: int(c.truth) for c in questions.generate_candidates(bundle, kinds=("execution_count",))}
        assert counts[2] == 1
        assert counts[3] == 6  # 5 iterations plus the exhausting check

    def test_branch_truth_is_next_line(self, bundle):
        branch = [c for c in questions.generate_candidates(bundle, kinds=("branch_at",)) if c.line == 4]
        assert {c.truth for c in branch} == {"5", "7"}

    def test_locality_is_inverse_of_steps(self, bundle):
        for cand in questions.generate_candidates(bundle):
            assert abs(cand.locality - 1 / cand.steps_needed) < 1e-9


class TestGrading:
    def test_last_tag_wins(self):
        assert grading.extract_answer("maybe <final>7</final> no wait <final>42</final>") == "42"

    def test_missing_tag_is_none(self):
        assert grading.extract_answer("I think 42") is None

    def test_soft_comparison(self):
        assert grading.is_correct(" 42 ", "42")
        assert grading.is_correct("[1, 2, 3]", "[1,2,3]")
        assert not grading.is_correct("41", "42")
        assert not grading.is_correct(None, "42")

    def test_escaped_newlines_in_answer_match_real_newlines(self):
        # hosted models write multi-line stdout as an escaped literal: backslash-n, not a newline
        assert grading.is_correct("\\n2\\n1 4\\n4 1\\n", "2\n1 4\n4 1\n")
        assert grading.is_correct("'0\\n0\\n'", "0\n0\n")

    def test_truth_is_never_unescaped(self):
        # a trace value that genuinely contains backslash-n must still require the same text
        assert grading.is_correct("'a\\nb'", "'a\\nb'")
        assert not grading.is_correct("a\nb", "'a\\nb'")

    def test_wrong_multiline_answer_still_wrong(self):
        assert not grading.is_correct("\\n2\\n2 3\\n", "2\n1 4\n")


IMPORT_SRC = """import math
import functools
def area(r):
    pi = math.pi
    return pi * r * r
"""

CRASHING_SRC = """def f(xs):
    acc = 0
    for x in xs:
        acc += x
    raise ValueError("boom")
"""


class TestWriteFiltering:
    def test_imported_modules_are_not_candidates(self):
        import active_investigator.tracing as tracing

        bundle = tracing.trace_code(IMPORT_SRC, '{"r": 2}', "area")
        names = {name for step in bundle.steps for name in step.writes}
        assert "math" not in names and "functools" not in names
        assert "pi" in names

    def test_crashed_program_has_no_return_cap_but_keeps_steps(self):
        import pyine.evals.correctness.types as ct

        import active_investigator.pickers as pickers
        import active_investigator.tracing as tracing

        bundle = tracing.trace_code(CRASHING_SRC, '{"xs": [1, 2, 3]}', "f")
        assert bundle.exception is not None
        assert bundle.return_step is None
        assert bundle.last_write_before_return() is not None
        record = ct.EvalRecord(
            sample_id="s",
            problem_id="p",
            attempt_index=0,
            model_output="<final>6</final>",
            final_answer="6",
            expected_output="6",
            label=True,
            code_type="original",
            tags=[],
            difficulty_score=None,
            record={"code": CRASHING_SRC, "inputs": '{"xs": [1, 2, 3]}', "entrypoint": "f"},
        )
        cands = questions.generate_candidates(bundle)
        picked = pickers.FixedLastWritePicker().pick(bundle, cands, record, k=1).questions
        assert len(picked) == 1 and picked[0].variable == "acc" and picked[0].truth == "6"


SCRIPT_SRC = """n = int(input())
total = 0
for _ in range(n):
    total += int(input())
print(total)
"""


class TestScripts:
    def test_string_none_entrypoint_means_script(self):
        import active_investigator.tracing as tracing

        assert tracing.normalize_entrypoint("None") is None
        assert tracing.normalize_entrypoint("") is None
        assert tracing.normalize_entrypoint("Solution.longest") == "Solution.longest"

    def test_stdin_script_traces_with_mocked_input(self):
        import active_investigator.tracing as tracing

        bundle = tracing.trace_record({"code": SCRIPT_SRC, "inputs": "3\n1\n2\n4\n", "entrypoint": "None"})
        assert bundle.exception is None
        totals = [s.writes["total"] for s in bundle.steps if "total" in s.writes]
        assert totals == ["0", "1", "3", "7"]
        assert bundle.last_write_before_return() is not None
