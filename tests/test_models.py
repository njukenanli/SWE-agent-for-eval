from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from sweagent import __version__
from sweagent.agent.models import GenericAPIModelConfig, get_model
from sweagent.exceptions import ModelConfigurationError
from sweagent.tools.parsing import Identity
from sweagent.tools.tools import ToolConfig
from sweagent.types import History


def test_litellm_mock():
    model = get_model(
        GenericAPIModelConfig(
            name="gpt-4o",
            completion_kwargs={"mock_response": "Hello, world!"},
            api_key=SecretStr("dummy_key"),
            top_p=None,
        ),
        ToolConfig(
            parse_function=Identity(),
        ),
    )
    assert model.query(History([{"role": "user", "content": "Hello, world!"}])) == {"message": "Hello, world!"}  # type: ignore


def _make_mock_response(content: str = "mock") -> MagicMock:
    """Create a minimal mock response matching litellm's ModelResponse shape."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    choice.token_ids = None
    choice.provider_specific_fields = {}
    choice.logprobs = None
    response = MagicMock()
    response.choices = [choice]
    response.prompt_token_ids = None
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    return response


def _make_vllm_mock_response() -> MagicMock:
    response = _make_mock_response("generated")
    response.prompt_token_ids = [11, 12, 13]
    response.choices[0].provider_specific_fields = {"token_ids": [21, 22]}
    response.choices[0].logprobs = MagicMock()
    response.choices[0].logprobs.content = [
        {"token": "generated", "logprob": -0.25, "top_logprobs": []},
        MagicMock(token=" output", logprob=-1.5, top_logprobs=[]),
    ]
    return response


def test_vllm_token_metadata_is_returned_with_model_output():
    model = get_model(
        GenericAPIModelConfig(
            name="my-qwen-model",
            max_input_tokens=0,
            per_instance_cost_limit=0,
            total_cost_limit=0,
        ),
        ToolConfig(),
    )

    with (
        patch("litellm.completion", return_value=_make_vllm_mock_response()),
        patch("litellm.utils.token_counter", return_value=1),
    ):
        output = model.query(History([{"role": "user", "content": "test"}]))

    assert output["input_token_ids"] == [11, 12, 13]  # type: ignore[index]
    assert output["output_token_ids"] == [21, 22]  # type: ignore[index]
    assert output["output_token_probabilities"] == [  # type: ignore[index]
        math.exp(-0.25),
        math.exp(-1.5),
    ]


@pytest.mark.parametrize("missing_field", ["token_ids", "probabilities"])
def test_vllm_missing_token_metadata_is_omitted(missing_field):
    response = _make_vllm_mock_response()
    if missing_field == "token_ids":
        response.prompt_token_ids = None
        response.choices[0].provider_specific_fields = {}
    else:
        response.choices[0].logprobs = None
    model = get_model(
        GenericAPIModelConfig(
            name="my-qwen-model",
            max_input_tokens=0,
            per_instance_cost_limit=0,
            total_cost_limit=0,
        ),
        ToolConfig(),
    )

    with (
        patch("litellm.completion", return_value=response),
        patch("litellm.utils.token_counter", return_value=1),
    ):
        output = model.query(History([{"role": "user", "content": "test"}]))

    if missing_field == "token_ids":
        assert "input_token_ids" not in output
        assert "output_token_ids" not in output
        assert "output_token_probabilities" not in output
    else:
        assert output["input_token_ids"] == [11, 12, 13]  # type: ignore[index]
        assert output["output_token_ids"] == [21, 22]  # type: ignore[index]
        assert "output_token_probabilities" not in output


@pytest.mark.parametrize(
    ("logprob_content", "error"),
    [
        ([{"logprob": -0.25}], "different number"),
        ([{"logprob": -0.25}, {"logprob": "invalid"}], "invalid log-probability"),
        ([{"logprob": -0.25}, {"logprob": 0.5}], "positive log-probability"),
    ],
)
def test_vllm_token_metadata_rejects_invalid_probabilities(logprob_content, error):
    response = _make_vllm_mock_response()
    response.choices[0].logprobs.content = logprob_content
    model = get_model(
        GenericAPIModelConfig(
            name="my-qwen-model",
            max_input_tokens=0,
            per_instance_cost_limit=0,
            total_cost_limit=0,
        ),
        ToolConfig(),
    )

    with (
        patch("litellm.completion", return_value=response),
        patch("litellm.utils.token_counter", return_value=1),
        pytest.raises(ModelConfigurationError, match=error),
    ):
        model.query(History([{"role": "user", "content": "test"}]))


def test_user_agent_header_default():
    """User-Agent header is added automatically when no extra_headers are set."""
    model = get_model(
        GenericAPIModelConfig(
            name="gpt-4o",
            api_key=SecretStr("dummy_key"),
            top_p=None,
            per_instance_cost_limit=0,
            total_cost_limit=0,
        ),
        ToolConfig(parse_function=Identity()),
    )
    mock_response = _make_mock_response()
    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        model.query(History([{"role": "user", "content": "test"}]))
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args
        extra_headers = call_kwargs.kwargs.get("extra_headers", {})
        assert "User-Agent" in extra_headers
        assert extra_headers["User-Agent"] == f"swe-agent/{__version__}"


def test_user_agent_header_preserves_existing():
    """User-Agent header is not overridden when already provided by the user."""
    custom_ua = "my-custom-agent/1.0"
    model = get_model(
        GenericAPIModelConfig(
            name="gpt-4o",
            completion_kwargs={"extra_headers": {"User-Agent": custom_ua}},
            api_key=SecretStr("dummy_key"),
            top_p=None,
            per_instance_cost_limit=0,
            total_cost_limit=0,
        ),
        ToolConfig(parse_function=Identity()),
    )
    mock_response = _make_mock_response()
    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        model.query(History([{"role": "user", "content": "test"}]))
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args
        extra_headers = call_kwargs.kwargs.get("extra_headers", {})
        assert extra_headers["User-Agent"] == custom_ua


def test_user_agent_header_with_other_extra_headers():
    """User-Agent header is added alongside other existing extra_headers."""
    model = get_model(
        GenericAPIModelConfig(
            name="gpt-4o",
            completion_kwargs={"extra_headers": {"X-Custom": "value"}},
            api_key=SecretStr("dummy_key"),
            top_p=None,
            per_instance_cost_limit=0,
            total_cost_limit=0,
        ),
        ToolConfig(parse_function=Identity()),
    )
    mock_response = _make_mock_response()
    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        model.query(History([{"role": "user", "content": "test"}]))
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args
        extra_headers = call_kwargs.kwargs.get("extra_headers", {})
        assert extra_headers["User-Agent"] == f"swe-agent/{__version__}"
        assert extra_headers["X-Custom"] == "value"


def test_vllm_request_uses_openai_provider_and_normalized_base_url(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "dummy")
    monkeypatch.setenv("LLM_API_BASE", "http://localhost:5001")
    model = get_model(
        GenericAPIModelConfig(
            name="my-qwen-model",
            max_input_tokens=0,
            per_instance_cost_limit=0,
            total_cost_limit=0,
        ),
        ToolConfig(),
    )
    mock_response = _make_mock_response()

    with (
        patch("litellm.completion", return_value=mock_response) as mock_completion,
        patch("litellm.utils.token_counter", return_value=1),
    ):
        model.query(History([{"role": "user", "content": "test"}]))

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "my-qwen-model"
    assert call_kwargs["base_url"] == "http://localhost:5001/v1"
    assert call_kwargs["custom_llm_provider"] == "openai"
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["tools"]
