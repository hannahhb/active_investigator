import active_investigator.pickers as pickers
import active_investigator.questions as questions
import active_investigator.scorer as scorer_mod


class TestFixedPicker:
    def test_picks_final_total(self, bundle, record):
        cands = questions.generate_candidates(bundle)
        picked = pickers.FixedLastWritePicker().pick(bundle, cands, record, k=1).questions
        assert len(picked) == 1
        assert picked[0].variable == "total"
        assert picked[0].truth == "10"

    def test_k_takes_most_recent_writes(self, bundle, record):
        cands = questions.generate_candidates(bundle)
        picked = pickers.FixedLastWritePicker().pick(bundle, cands, record, k=3).questions
        assert [q.step for q in picked] == sorted((q.step for q in picked), reverse=True)


class TestPromptedPicker:
    def test_parses_json_indices(self, bundle, record, fake_responder_factory):
        class Chooser(fake_responder_factory):
            def ask(self, system, user):
                self.calls.append(user)
                return pickers.responder_mod.Reply(text="[2, 1]", completion_tokens=3.0)

        cands = questions.generate_candidates(bundle)
        result = pickers.PromptedPicker(Chooser(), min_steps=1).pick(bundle, cands, record, k=2)
        assert [q.text for q in result.questions] == [cands[1].text, cands[0].text]
        assert result.probe_tokens == 3.0

    def test_listing_is_question_text_only(self, bundle, record, fake_responder_factory):
        chooser = fake_responder_factory()
        cands = questions.generate_candidates(bundle)
        pickers.PromptedPicker(chooser, max_candidates_shown=10, min_steps=1).pick(bundle, cands, record, k=1)
        listing = chooser.calls[0].split("CANDIDATE QUESTIONS:\n")[1].split("\n\nChoose")[0]
        expected = "\n".join(f"{idx + 1}. {q.text}" for idx, q in enumerate(cands[:10]))
        assert listing == expected  # numbered question texts and nothing else: no truths leak


class TestPromptedDepthFloor:
    def test_single_line_questions_hidden_by_default(self, bundle, record, fake_responder_factory):
        chooser = fake_responder_factory()
        cands = questions.generate_candidates(bundle)
        pickers.PromptedPicker(chooser).pick(bundle, cands, record, k=1)
        listing = chooser.calls[0].split("CANDIDATE QUESTIONS:\n")[1].split("\n\nChoose")[0]
        trivial = [q.text for q in cands if q.steps_needed == 1]
        assert trivial and not any(text in listing for text in trivial)


class TestRankedPicker:
    def test_wrong_answers_rank_first_by_locality(self, bundle, record, fake_responder_factory):
        cands = questions.generate_candidates(bundle, kinds=("value_at",))
        # the responder gets every question about `n` right (cheap index reads) and everything else wrong
        known = {c.text: c.truth for c in cands if c.variable == "n"}

        class KnowsN(fake_responder_factory):
            def ask(self, system, user):
                self.calls.append(user)
                for text, truth in known.items():
                    if text in user:
                        return pickers.responder_mod.Reply(text=f"<final>{truth}</final>", completion_tokens=2.0)
                return pickers.responder_mod.Reply(text="<final>wrong</final>", completion_tokens=2.0)

        result = pickers.RankedPicker(KnowsN(), max_probe=8).pick(bundle, cands, record, k=3)
        assert result.notes["probed"] == 8
        assert len(result.prefetched) == 8
        assert result.probe_tokens == 16.0
        picked = result.questions
        assert all(q.variable != "n" for q in picked), "correctly answered questions must rank last"
        localities = [q.locality for q in picked]
        assert localities == sorted(localities, reverse=True)


class TestScorer:
    def test_scores_costs_and_metadata_align(self, bundle, record, fake_responder_factory):
        responder = fake_responder_factory(answers={"value of `total`": "10"})
        scorer = scorer_mod.InvestigatorScorer(responder, pickers.FixedLastWritePicker(), k=1, max_workers=2)
        result = scorer.score_records([record, record])
        assert result.scores == [1.0, 1.0]
        assert result.verification_costs == [5.0, 5.0]
        assert len(result.attempt_metadata) == 2
        asked = next(iter(result.attempt_metadata.values()))["asked"][0]
        assert asked["correct"] and asked["truth"] == "10"

    def test_wrong_answer_scores_zero(self, bundle, record, fake_responder_factory):
        scorer = scorer_mod.InvestigatorScorer(fake_responder_factory(), pickers.FixedLastWritePicker(), k=1)
        assert scorer.score_records([record]).scores == [0.0]

    def test_untraceable_program_gets_default_score(self, record, fake_responder_factory):
        broken = record.__class__(**{**record.__dict__, "record": {**record.record, "code": "def f(:\n  pass"}})
        scorer = scorer_mod.InvestigatorScorer(fake_responder_factory(), pickers.FixedLastWritePicker())
        result = scorer.score_records([broken])
        assert result.scores == [0.5]
        assert next(iter(result.attempt_metadata.values()))["outcome"] == "trace_failed"
        assert scorer.counters["trace_failed"] == 1


class TestPromptRendering:
    def test_program_is_shown_with_line_numbers(self, bundle, record):
        cands = questions.generate_candidates(bundle)
        prompt = pickers.render_followup(pickers.responder_mod.load_prompt("followup_user.txt"), record, cands[0])
        assert "   1| def process(nums):" in prompt
        assert "   2|     total = 0" in prompt


class TestTracingHappensBeforeThePool:
    def test_all_tracing_on_main_thread_before_any_model_call(self, record, fake_responder_factory, monkeypatch):
        import threading

        import active_investigator.tracing as tracing

        events: list[str] = []
        real = tracing.trace_record

        def traced(payload, timeout_seconds=30):
            events.append("trace:" + threading.current_thread().name)
            return real(payload, timeout_seconds=timeout_seconds)

        responder = fake_responder_factory(answers={"value of `total`": "10"})
        original_ask = responder.ask

        def ask(system, user):
            events.append("ask")
            return original_ask(system, user)

        responder.ask = ask
        monkeypatch.setattr(scorer_mod.tracing, "trace_record", traced)
        scorer = scorer_mod.InvestigatorScorer(responder, pickers.FixedLastWritePicker(), k=1, max_workers=4)
        result = scorer.score_records([record, record, record])
        assert result.scores == [1.0, 1.0, 1.0]
        traces = [e for e in events if e.startswith("trace:")]
        assert len(traces) == 3
        assert all(e == "trace:MainThread" for e in traces), traces
        assert events.index("ask") > max(i for i, e in enumerate(events) if e.startswith("trace:"))
