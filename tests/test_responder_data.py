import pytest

import active_investigator.data as data
import active_investigator.responder as responder_mod


class TestProviderSpec:
    def test_vllm_defaults_to_local_server(self):
        spec = responder_mod.ProviderSpec.parse("vllm:pyine-organism")
        assert (spec.provider, spec.model, spec.base_url) == ("vllm", "pyine-organism", "http://localhost:8000/v1")

    def test_vllm_explicit_url(self):
        spec = responder_mod.ProviderSpec.parse("vllm:pyine-organism@http://gpu-node:8001/v1")
        assert spec.base_url == "http://gpu-node:8001/v1"

    def test_openrouter_is_openai_protocol_at_openrouter_url(self):
        spec = responder_mod.ProviderSpec.parse("openrouter:meta-llama/llama-3.3-70b-instruct")
        assert spec.provider == "openai"
        assert spec.model == "meta-llama/llama-3.3-70b-instruct"
        assert spec.base_url == responder_mod.OPENROUTER_BASE_URL

    def test_openai_has_no_base_url_override(self):
        spec = responder_mod.ProviderSpec.parse("openai:gpt-4.1-mini")
        assert spec.base_url is None

    @pytest.mark.parametrize("bad", ["gpt-4.1-mini", "vllm:", "ollama:llama3"])
    def test_bad_specs_raise(self, bad):
        with pytest.raises(ValueError):
            responder_mod.ProviderSpec.parse(bad)


class TestRegeneratePredictions:
    @pytest.fixture
    def captured_chain(self, monkeypatch):
        calls: list[dict] = []

        class FakeChain:
            def invoke(self, variables, config=None):
                calls.append(variables)
                return "tracing it... <final>10</final>"

        monkeypatch.setattr(data.pyine.prompts.manager, "get_prompt_chain", lambda **kwargs: FakeChain())
        return calls

    def test_passes_every_template_variable(self, record, fake_responder_factory, captured_chain):
        record.record.update({"first_line": 7, "first_line_hit": 2, "last_line": 9, "last_line_hit": 1})
        data.regenerate_predictions([record], fake_responder_factory())
        variables = captured_chain[0]
        for key in (
            "code",
            "predict_type",
            "entrypoint",
            "inputs",
            "first_line",
            "first_line_hit",
            "last_line",
            "last_line_hit",
        ):
            assert key in variables, key
        assert variables["first_line"] == 7 and variables["last_line_hit"] == 1

    def test_missing_line_fields_default_rather_than_fail(self, record, fake_responder_factory, captured_chain):
        data.regenerate_predictions([record], fake_responder_factory())
        assert captured_chain[0]["first_line"] == 0

    def test_relabels_against_expected_output(self, record, fake_responder_factory, captured_chain):
        (out,) = data.regenerate_predictions([record], fake_responder_factory())
        assert out.final_answer == "10"
        assert out.label is True
        assert out.record["model_output"].endswith("<final>10</final>")


class TestProblemSubset:
    def test_max_problems_keeps_whole_problems(self, monkeypatch, record):
        import pyine.evals.correctness.types as ct

        def make(problem, sample):
            return ct.EvalRecord(**{**record.__dict__, "problem_id": problem, "sample_id": f"{problem}/{sample}"})

        pool = [make(f"P{p}", s) for p in range(5) for s in range(3)]

        class FakeDM:
            def prepare_data(self): ...

            def setup(self): ...

            def get_records_for_subset(self, subset):
                return list(pool)

        class FakeConfig:
            def __init__(self, **kwargs): ...

            def instantiate_datamodule(self):
                return FakeDM()

        monkeypatch.setattr(data.datamodule_configs, "CorrectnessDataModuleConfig", FakeConfig)
        kept = data.load_records("x", lmdb_paths=["x"], max_problems=2, seed=0)
        assert len({r.problem_id for r in kept}) == 2
        assert len(kept) == 6  # three samples per problem, none dropped


class TestCliDefaults:
    def test_default_split_is_full_taco_and_problems_has_short_flag(self):
        import active_investigator.cli as cli

        parser = cli.build_parser()
        args = parser.parse_args(["run", "--arm", "fixed", "--responder", "vllm:x", "-p", "40"])
        assert args.split == "TACO"
        assert args.problems == 40
