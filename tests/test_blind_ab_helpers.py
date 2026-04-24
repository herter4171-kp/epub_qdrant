"""Unit tests for mcp_call and llm_call helpers in blind_ab_test.py."""

from unittest.mock import patch, MagicMock

import pytest

from scripts.blind_ab_test import mcp_call, llm_call, ANSWER_MODEL, ANSWER_TEMPERATURE


# ─── mcp_call ────────────────────────────────────────────────────────


class TestMcpCall:
    """Test mcp_call JSON-RPC format and response parsing."""

    def _mock_response(self, json_body, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_body
        resp.raise_for_status.return_value = None
        return resp

    @patch("scripts.blind_ab_test.requests.post")
    def test_sends_jsonrpc_format(self, mock_post):
        """Verify JSON-RPC 2.0 envelope sent to MCP."""
        mock_post.return_value = self._mock_response({"result": {"foo": "bar"}})
        mcp_call("test_tool", {"key": "val"})
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        assert body["jsonrpc"] == "2.0"
        assert body["method"] == "tools/call"
        assert body["params"]["name"] == "test_tool"
        assert body["params"]["arguments"] == {"key": "val"}

    @patch("scripts.blind_ab_test.requests.post")
    def test_direct_dict_response(self, mock_post):
        """Shape 1: result is a plain dict without 'content' key."""
        mock_post.return_value = self._mock_response({"result": {"data": 42}})
        result = mcp_call("tool", {})
        assert result == {"data": 42}

    @patch("scripts.blind_ab_test.requests.post")
    def test_content_wrapper_response(self, mock_post):
        """Shape 2: result.content[0].text with JSON string."""
        mock_post.return_value = self._mock_response({
            "result": {
                "content": [{"text": '{"books": [{"title": "RAG"}]}'}]
            }
        })
        result = mcp_call("tool", {})
        assert result == {"books": [{"title": "RAG"}]}

    @patch("scripts.blind_ab_test.requests.post")
    def test_content_wrapper_non_json_text(self, mock_post):
        """Shape 2 fallback: content text is not valid JSON."""
        mock_post.return_value = self._mock_response({
            "result": {"content": [{"text": "plain string"}]}
        })
        result = mcp_call("tool", {})
        assert result == {"raw_text": "plain string"}

    @patch("scripts.blind_ab_test.requests.post")
    def test_fallback_response(self, mock_post):
        """Shape 3: result has content key but empty list."""
        mock_post.return_value = self._mock_response({
            "result": {"content": []}
        })
        result = mcp_call("tool", {})
        assert result == {"content": []}

    @patch("scripts.blind_ab_test.requests.post")
    def test_error_in_body_raises(self, mock_post):
        """MCP error response raises RuntimeError."""
        mock_post.return_value = self._mock_response({
            "error": {"message": "tool not found"}
        })
        with pytest.raises(RuntimeError, match="tool not found"):
            mcp_call("bad_tool", {})


# ─── llm_call ────────────────────────────────────────────────────────


class TestLlmCall:
    """Test llm_call OpenAI format and parameter passing."""

    def _mock_response(self, content="test response"):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        resp.raise_for_status.return_value = None
        return resp

    @patch("scripts.blind_ab_test.requests.post")
    def test_sends_openai_format(self, mock_post):
        """Verify OpenAI chat completions envelope."""
        mock_post.return_value = self._mock_response()
        llm_call("sys prompt", "user msg")
        body = mock_post.call_args.kwargs["json"]
        assert body["messages"][0] == {"role": "system", "content": "sys prompt"}
        assert body["messages"][1] == {"role": "user", "content": "user msg"}

    @patch("scripts.blind_ab_test.requests.post")
    def test_returns_content_string(self, mock_post):
        """Returns message content string."""
        mock_post.return_value = self._mock_response("the answer")
        result = llm_call("sys", "user")
        assert result == "the answer"

    @patch("scripts.blind_ab_test.requests.post")
    def test_default_model_and_temperature(self, mock_post):
        """Uses ANSWER_MODEL and ANSWER_TEMPERATURE when not specified."""
        mock_post.return_value = self._mock_response()
        llm_call("sys", "user")
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == ANSWER_MODEL
        assert body["temperature"] == ANSWER_TEMPERATURE

    @patch("scripts.blind_ab_test.requests.post")
    def test_custom_model_and_temperature(self, mock_post):
        """Passes explicit model and temperature when provided."""
        mock_post.return_value = self._mock_response()
        llm_call("sys", "user", model="gpt-4", temperature=0.7)
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "gpt-4"
        assert body["temperature"] == 0.7

    @patch("scripts.blind_ab_test.requests.post")
    def test_zero_temperature_passed(self, mock_post):
        """temperature=0.0 should be passed, not replaced by default."""
        mock_post.return_value = self._mock_response()
        llm_call("sys", "user", temperature=0.0)
        body = mock_post.call_args.kwargs["json"]
        assert body["temperature"] == 0.0
