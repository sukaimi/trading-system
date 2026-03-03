"""Tests for core/llm_client.py"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from core.llm_client import LLMClient


@pytest.fixture
def mock_client():
    return LLMClient(mock_mode=True)


@pytest.fixture
def real_client():
    return LLMClient(mock_mode=False)


class SampleSchema(BaseModel):
    name: str = "test"
    value: int = 42


class TestMockMode:
    def test_auto_detect_no_keys(self, mock_client):
        assert mock_client.mock_mode is True

    def test_explicit_mock(self):
        c = LLMClient(mock_mode=True)
        assert c.mock_mode is True

    def test_explicit_real(self):
        c = LLMClient(mock_mode=False)
        assert c.mock_mode is False


class TestMockResponses:
    def test_deepseek_mock(self, mock_client):
        result = mock_client.call_deepseek("test prompt")
        assert isinstance(result, dict)
        assert result.get("mock") is True

    def test_kimi_mock(self, mock_client):
        result = mock_client.call_kimi("test prompt")
        assert isinstance(result, dict)
        assert result.get("mock") is True

    def test_anthropic_mock(self, mock_client):
        result = mock_client.call_anthropic("test prompt")
        assert isinstance(result, dict)
        assert result.get("mock") is True

    def test_mock_with_schema(self, mock_client):
        result = mock_client.call_deepseek("test", response_schema=SampleSchema)
        assert isinstance(result, dict)


class TestParseJSON:
    def test_plain_json(self, mock_client):
        raw = '{"key": "value"}'
        result = mock_client._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_markdown_code_block(self, mock_client):
        raw = '```json\n{"key": "value"}\n```'
        result = mock_client._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_embedded_json(self, mock_client):
        raw = 'Here is the result: {"key": "value"} and more text'
        result = mock_client._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_invalid_json(self, mock_client):
        raw = "not json at all"
        result = mock_client._parse_json_response(raw)
        assert "error" in result

    def test_json_array(self, mock_client):
        raw = '[{"a": 1}, {"b": 2}]'
        result = mock_client._parse_json_response(raw)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_schema_validation(self, mock_client):
        raw = '{"name": "hello", "value": 99}'
        result = mock_client._parse_json_response(raw, schema=SampleSchema)
        assert result["name"] == "hello"
        assert result["value"] == 99


class TestBudgetCheck:
    def test_anthropic_blocked_when_budget_exceeded(self, real_client):
        mock_tracker = MagicMock()
        mock_tracker.check_budget.return_value = False
        real_client.set_cost_tracker(mock_tracker)

        result = real_client.call_anthropic("test prompt")
        assert result == {"error": "daily_budget_exceeded"}
        mock_tracker.check_budget.assert_called_once_with("anthropic")

    def test_anthropic_proceeds_when_budget_ok(self, mock_client):
        mock_tracker = MagicMock()
        mock_tracker.check_budget.return_value = True
        mock_client.set_cost_tracker(mock_tracker)

        # Mock mode returns mock response, so budget check is skipped
        result = mock_client.call_anthropic("test prompt")
        assert result.get("mock") is True

    def test_max_tokens_parameter(self, mock_client):
        # Should accept max_tokens without error
        result = mock_client.call_anthropic("hi", "reply", max_tokens=5)
        assert isinstance(result, dict)


class TestCallWithFallback:
    def test_mock_mode(self, mock_client):
        result = mock_client.call_with_fallback("prompt")
        assert isinstance(result, dict)

    @patch.object(LLMClient, "call_deepseek", return_value={"error": "fail"})
    @patch.object(LLMClient, "_call_gemini", side_effect=Exception("gemini fail"))
    def test_fallback_chain(self, mock_gemini, mock_ds, real_client):
        # Both fail — retries deepseek
        result = real_client.call_with_fallback("prompt")
        assert mock_ds.call_count == 2
