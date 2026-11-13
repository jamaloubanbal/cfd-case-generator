"""
Tests for app/llm.py — explain_case() function.

No real API calls are made; Anthropic client and requests are mocked.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.llm import explain_case, _extract_json


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ANALYSIS = {
    "solver": "simpleFoam",
    "solver_justification": "Steady incompressible flow.",
    "fluid": {"name": "air", "nu": 1.5e-5, "rho": 1.225},
    "turb_model": "kEpsilon",
    "geometry_type": "channel",
    "Re": 10000.0,
    "is_transient": False,
    "boundary_conditions": [
        {"patch_name": "inlet",  "type": "fixedValue", "values": {"U": [1.0, 0.0, 0.0]}},
        {"patch_name": "outlet", "type": "zeroGradient", "values": {}},
    ],
    "warnings": [],
}

SAMPLE_DESCRIPTION = "Air flow in a rectangular channel at Re=10000."

MOCK_EXPLANATION = """\
**Flow regime & physics**
Turbulent internal flow at Re=10 000.

**Expected flow features**
Fully developed turbulent boundary layers on the top and bottom walls.

**Convergence tips**
Watch the p residual. Expect convergence within 500 iterations.

**Post-processing checklist**
Plot velocity profiles and wall shear stress.
"""


# ─────────────────────────────────────────────────────────────────────────────
# _extract_json helper
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_clean_json(self):
        raw = '{"solver": "simpleFoam"}'
        assert _extract_json(raw) == {"solver": "simpleFoam"}

    def test_strips_markdown_fence(self):
        raw = "```json\n{\"solver\": \"simpleFoam\"}\n```"
        assert _extract_json(raw) == {"solver": "simpleFoam"}

    def test_strips_plain_fence(self):
        raw = "```\n{\"a\": 1}\n```"
        assert _extract_json(raw) == {"a": 1}

    def test_invalid_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("not json")


# ─────────────────────────────────────────────────────────────────────────────
# explain_case — Claude backend
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainCaseClaude:

    def _mock_claude_response(self, text: str):
        """Build a mock Messages API response (a thinking block, then text)."""
        thinking_block = MagicMock(type="thinking", thinking="")
        content_block = MagicMock(type="text", text=text)
        message = MagicMock(stop_reason="end_turn")
        message.content = [thinking_block, content_block]
        return message

    @staticmethod
    def _stub(client, response=None, side_effect=None):
        """Stub both the GA and the beta (refusal-fallback) create endpoints."""
        for create in (client.messages.create, client.beta.messages.create):
            create.return_value = response
            create.side_effect = side_effect

    @staticmethod
    def _call_kwargs(client) -> dict:
        """kwargs of whichever create endpoint was called."""
        for create in (client.beta.messages.create, client.messages.create):
            if create.called:
                return create.call_args.kwargs
        raise AssertionError("no Messages API call was made")

    def test_returns_string(self):
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            self._stub(client, self._mock_claude_response(MOCK_EXPLANATION))
            result = explain_case(
                SAMPLE_ANALYSIS,
                SAMPLE_DESCRIPTION,
                backend="claude",
                model="claude-sonnet-4-6",
                api_key="sk-test",
            )
        assert isinstance(result, str)
        assert len(result) > 20

    def test_strips_whitespace(self):
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            self._stub(client, self._mock_claude_response(
                "  \n" + MOCK_EXPLANATION + "\n  "
            ))
            result = explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="claude", api_key="sk-test"
            )
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_passes_api_key(self):
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            self._stub(client, self._mock_claude_response(MOCK_EXPLANATION))
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="claude", api_key="my-key"
            )
        mock_cls.assert_called_once_with(api_key="my-key")

    def test_uses_correct_model(self):
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            self._stub(client, self._mock_claude_response(MOCK_EXPLANATION))
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION,
                backend="claude", model="claude-opus-4-6", api_key="sk-test"
            )
        assert self._call_kwargs(client)["model"] == "claude-opus-4-6"

    def test_no_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("os.environ.get", return_value=None):
                with pytest.raises(RuntimeError, match="API key"):
                    explain_case(
                        SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION,
                        backend="claude", api_key=None
                    )

    def test_analysis_json_included_in_prompt(self):
        """The analysis dict should be serialised into the user message."""
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            self._stub(client, self._mock_claude_response(MOCK_EXPLANATION))
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="claude", api_key="sk-test"
            )
        call_kwargs = self._call_kwargs(client)
        user_content = call_kwargs["messages"][0]["content"]
        assert "simpleFoam" in user_content
        assert SAMPLE_DESCRIPTION in user_content

    def test_max_tokens_limited(self):
        """explain_case requests far fewer tokens than analyze_cfd_problem (16 000)."""
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            self._stub(client, self._mock_claude_response(MOCK_EXPLANATION))
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="claude", api_key="sk-test"
            )
        assert self._call_kwargs(client)["max_tokens"] <= 4096

    def test_anthropic_api_error_propagates(self):
        import anthropic as _anthropic
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            self._stub(client, side_effect=_anthropic.APIError(
                message="rate limit", request=MagicMock(), body=None
            ))
            with pytest.raises(Exception):
                explain_case(
                    SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="claude", api_key="sk-test"
                )


# ─────────────────────────────────────────────────────────────────────────────
# explain_case — Ollama backend
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainCaseOllama:

    def _mock_ollama_resp(self, text: str):
        resp = MagicMock()
        resp.json.return_value = {"response": text}
        resp.raise_for_status.return_value = None
        return resp

    def test_returns_string(self):
        with patch("app.llm.requests.post") as mock_post:
            mock_post.return_value = self._mock_ollama_resp(MOCK_EXPLANATION)
            result = explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION,
                backend="ollama", model="mistral",
            )
        assert isinstance(result, str)
        assert len(result) > 10

    def test_posts_to_generate_endpoint(self):
        with patch("app.llm.requests.post") as mock_post:
            mock_post.return_value = self._mock_ollama_resp(MOCK_EXPLANATION)
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION,
                backend="ollama", model="mistral",
                ollama_base_url="http://localhost:11434",
            )
        url = mock_post.call_args.args[0]
        assert url.endswith("/api/generate")

    def test_uses_specified_model(self):
        with patch("app.llm.requests.post") as mock_post:
            mock_post.return_value = self._mock_ollama_resp(MOCK_EXPLANATION)
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION,
                backend="ollama", model="llama3",
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "llama3"

    def test_timeout_passed(self):
        with patch("app.llm.requests.post") as mock_post:
            mock_post.return_value = self._mock_ollama_resp(MOCK_EXPLANATION)
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION,
                backend="ollama", ollama_timeout=60,
            )
        assert mock_post.call_args.kwargs["timeout"] == 60

    def test_http_error_propagates(self):
        import requests as _req
        with patch("app.llm.requests.post") as mock_post:
            resp = MagicMock()
            resp.raise_for_status.side_effect = _req.exceptions.HTTPError("503")
            mock_post.return_value = resp
            with pytest.raises(_req.exceptions.HTTPError):
                explain_case(
                    SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="ollama"
                )


# ─────────────────────────────────────────────────────────────────────────────
# explain_case — unknown backend
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainCaseUnknownBackend:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            explain_case(SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="gpt4")


# ─────────────────────────────────────────────────────────────────────────────
# analyze_cfd_problem — structured outputs, fallbacks, refusals
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeStructuredOutput:

    @staticmethod
    def _message(text: str, stop_reason: str = "end_turn"):
        message = MagicMock(stop_reason=stop_reason)
        message.content = [MagicMock(type="thinking", thinking=""), MagicMock(type="text", text=text)]
        return message

    def test_default_model_uses_schema_and_refusal_fallbacks(self):
        from app.llm import DEFAULT_CLAUDE_MODEL, analyze_cfd_problem
        from app.schema import ANALYSIS_SCHEMA
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            client.beta.messages.create.return_value = self._message(json.dumps(SAMPLE_ANALYSIS))
            result = analyze_cfd_problem("air in a duct", backend="claude", api_key="sk-test")
        kwargs = client.beta.messages.create.call_args.kwargs
        assert kwargs["model"] == DEFAULT_CLAUDE_MODEL
        assert kwargs["fallbacks"] == "default"
        assert kwargs["output_config"]["format"] == {"type": "json_schema", "schema": ANALYSIS_SCHEMA}
        assert result["solver"] == "simpleFoam"

    def test_other_models_use_ga_endpoint_without_fallbacks(self):
        from app.llm import analyze_cfd_problem
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            client.messages.create.return_value = self._message(json.dumps(SAMPLE_ANALYSIS))
            analyze_cfd_problem("air", backend="claude", model="claude-haiku-4-5", api_key="sk")
        kwargs = client.messages.create.call_args.kwargs
        assert "fallbacks" not in kwargs
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        client.beta.messages.create.assert_not_called()

    def test_refusal_raises_clear_error(self):
        from app.llm import analyze_cfd_problem
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            client.beta.messages.create.return_value = self._message("", stop_reason="refusal")
            with pytest.raises(RuntimeError, match="declined"):
                analyze_cfd_problem("x", backend="claude", api_key="sk-test")

    def test_truncation_raises_clear_error(self):
        from app.llm import analyze_cfd_problem
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            client.beta.messages.create.return_value = self._message('{"solv', stop_reason="max_tokens")
            with pytest.raises(RuntimeError, match="cut off"):
                analyze_cfd_problem("x", backend="claude", api_key="sk-test")

    def test_ollama_sends_schema_as_format(self):
        from app.llm import analyze_cfd_problem
        from app.schema import ANALYSIS_SCHEMA
        resp = MagicMock()
        resp.json.return_value = {"response": json.dumps(SAMPLE_ANALYSIS)}
        with patch("app.llm.requests.post", return_value=resp) as mock_post:
            analyze_cfd_problem("x", backend="ollama", model="mistral")
        assert mock_post.call_args.kwargs["json"]["format"] == ANALYSIS_SCHEMA


class TestAnalysisSchema:
    """The schema must satisfy the structured-output rules."""

    def _walk(self, node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from self._walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from self._walk(v)

    def test_every_object_is_strict(self):
        from app.schema import ANALYSIS_SCHEMA
        for obj in self._walk(ANALYSIS_SCHEMA):
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"])

    def test_no_unsupported_numeric_constraints(self):
        from app.schema import ANALYSIS_SCHEMA
        text = json.dumps(ANALYSIS_SCHEMA)
        for kw in ("minimum", "maximum", "minLength", "maxLength", "multipleOf"):
            assert f'"{kw}"' not in text

    def test_enums_match_generators(self):
        from app.generators import GENERATORS
        from app.generators.base import GEOMETRY_PATCHES
        from app.schema import GEOMETRIES, SOLVERS
        assert set(SOLVERS) == set(GENERATORS)
        assert set(GEOMETRIES) == set(GEOMETRY_PATCHES) - {"stl_import"}


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible backends (Gemini, Groq, OpenRouter, …) — mocked HTTP
# ─────────────────────────────────────────────────────────────────────────────

def _chat_response(content: str, status: int = 200):
    import requests as _rq
    resp = MagicMock(status_code=status, text="bad request")
    resp.json.return_value = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if status >= 400:
        resp.raise_for_status.side_effect = _rq.exceptions.HTTPError(response=resp)
    return resp


def test_gemini_sends_strict_schema_and_bearer_key():
    from app.llm import analyze_cfd_problem
    from app.schema import ANALYSIS_SCHEMA
    with patch("app.llm.requests.post", return_value=_chat_response(json.dumps(SAMPLE_ANALYSIS))) as post:
        result = analyze_cfd_problem("air in a duct", backend="gemini", api_key="AIza-test")
    url, kwargs = post.call_args.args[0], post.call_args.kwargs
    assert url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer AIza-test"
    assert kwargs["json"]["response_format"]["json_schema"]["schema"] == ANALYSIS_SCHEMA
    assert result["solver"] == "simpleFoam"


def test_provider_without_json_schema_falls_back_to_json_mode():
    """Some providers reject response_format=json_schema with 400 → JSON mode, schema in the prompt."""
    from app.llm import analyze_cfd_problem
    responses = [_chat_response("", status=400), _chat_response(json.dumps(SAMPLE_ANALYSIS))]
    with patch("app.llm.requests.post", side_effect=responses) as post:
        result = analyze_cfd_problem("air in a duct", backend="groq", api_key="gsk-test")
    second = post.call_args_list[1].kwargs["json"]
    assert second["response_format"] == {"type": "json_object"}
    assert '"additionalProperties"' in second["messages"][0]["content"]      # schema in the prompt
    assert result["solver"] == "simpleFoam"
