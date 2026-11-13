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
        """Build a mock Anthropic messages.create response."""
        content_block = MagicMock()
        content_block.text = text
        message = MagicMock()
        message.content = [content_block]
        return message

    def test_returns_string(self):
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            client.messages.create.return_value = self._mock_claude_response(MOCK_EXPLANATION)
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
            client.messages.create.return_value = self._mock_claude_response(
                "  \n" + MOCK_EXPLANATION + "\n  "
            )
            result = explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="claude", api_key="sk-test"
            )
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_passes_api_key(self):
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            client.messages.create.return_value = self._mock_claude_response(MOCK_EXPLANATION)
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="claude", api_key="my-key"
            )
        mock_cls.assert_called_once_with(api_key="my-key")

    def test_uses_correct_model(self):
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            client.messages.create.return_value = self._mock_claude_response(MOCK_EXPLANATION)
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION,
                backend="claude", model="claude-opus-4-6", api_key="sk-test"
            )
        call_kwargs = client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-opus-4-6"

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
            client.messages.create.return_value = self._mock_claude_response(MOCK_EXPLANATION)
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="claude", api_key="sk-test"
            )
        call_kwargs = client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "simpleFoam" in user_content
        assert SAMPLE_DESCRIPTION in user_content

    def test_max_tokens_limited(self):
        """explain_case should request far fewer tokens than analyze_cfd_problem."""
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            client.messages.create.return_value = self._mock_claude_response(MOCK_EXPLANATION)
            explain_case(
                SAMPLE_ANALYSIS, SAMPLE_DESCRIPTION, backend="claude", api_key="sk-test"
            )
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] <= 1024

    def test_anthropic_api_error_propagates(self):
        import anthropic as _anthropic
        with patch("app.llm.anthropic.Anthropic") as mock_cls:
            client = mock_cls.return_value
            client.messages.create.side_effect = _anthropic.APIError(
                message="rate limit", request=MagicMock(), body=None
            )
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
