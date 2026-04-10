from unittest.mock import patch

import pytest

from mem0.configs.llms.codex import CodexConfig
from mem0.llms.codex import CodexLLM


@pytest.fixture
def mock_codex():
    with patch("mem0.llms.codex.CodexLLM._run_codex") as mock_run:
        mock_run.return_value = "Hello, world!"
        yield mock_run


def test_default_model():
    config = CodexConfig()
    llm = CodexLLM(config)
    assert llm.config.model == "o4-mini"


def test_custom_model():
    config = CodexConfig(model="gpt-5")
    llm = CodexLLM(config)
    assert llm.config.model == "gpt-5"


def test_codex_config_defaults():
    config = CodexConfig()
    assert config.codex_cli_path is None
    assert config.full_context is False
    assert config.temperature == 0.1
    assert config.max_tokens == 2000


def test_codex_config_custom():
    config = CodexConfig(codex_cli_path="/usr/local/bin/codex", full_context=True)
    assert config.codex_cli_path == "/usr/local/bin/codex"
    assert config.full_context is True


def test_format_messages_single_user():
    llm = CodexLLM(CodexConfig(model="o4-mini"))
    messages = [{"role": "user", "content": "Hello"}]
    result = llm._format_messages(messages)
    assert result == "User: Hello"


def test_format_messages_system_and_user():
    llm = CodexLLM(CodexConfig(model="o4-mini"))
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    result = llm._format_messages(messages)
    assert result == "System: You are a helpful assistant.\n\nUser: Hello, how are you?"


def test_format_messages_with_assistant():
    llm = CodexLLM(CodexConfig(model="o4-mini"))
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    result = llm._format_messages(messages)
    assert "System: Be helpful." in result
    assert "User: Hi" in result
    assert "Assistant: Hello!" in result
    assert "User: How are you?" in result


def test_generate_response_without_tools(mock_codex):
    config = CodexConfig(model="o4-mini")
    llm = CodexLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    mock_codex.return_value = "I'm doing well, thank you for asking!"

    response = llm.generate_response(messages)

    mock_codex.assert_called_once()
    assert response == "I'm doing well, thank you for asking!"


def test_generate_response_with_json_format(mock_codex):
    config = CodexConfig(model="o4-mini")
    llm = CodexLLM(config)
    messages = [
        {"role": "system", "content": "Extract facts."},
        {"role": "user", "content": "The sky is blue."},
    ]

    mock_codex.return_value = '{"facts": ["The sky is blue"]}'

    response = llm.generate_response(messages, response_format={"type": "json_object"})

    # Verify the prompt included JSON instruction
    call_args = mock_codex.call_args[0][0]
    assert "You must respond with valid JSON only." in call_args
    assert response == '{"facts": ["The sky is blue"]}'


def test_generate_response_with_tools(mock_codex):
    config = CodexConfig(model="o4-mini")
    llm = CodexLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Add a new memory: Today is a sunny day."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_memory",
                "description": "Add a memory",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "string", "description": "Data to add to memory"}},
                    "required": ["data"],
                },
            },
        }
    ]

    mock_codex.return_value = (
        '{"tool_calls": [{"name": "add_memory", "arguments": {"data": "Today is a sunny day."}}], '
        '"content": "I\'ve added the memory for you."}'
    )

    response = llm.generate_response(messages, tools=tools)

    assert response["content"] == "I've added the memory for you."
    assert len(response["tool_calls"]) == 1
    assert response["tool_calls"][0]["name"] == "add_memory"
    assert response["tool_calls"][0]["arguments"] == {"data": "Today is a sunny day."}


def test_generate_response_tool_parse_failure(mock_codex):
    config = CodexConfig(model="o4-mini")
    llm = CodexLLM(config)
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool", "description": "A test tool", "parameters": {}}}]

    mock_codex.return_value = "This is not valid JSON for tool calls"

    response = llm.generate_response(messages, tools=tools)

    assert response["content"] == "This is not valid JSON for tool calls"
    assert response["tool_calls"] == []


def test_format_tools_prompt():
    llm = CodexLLM(CodexConfig(model="o4-mini"))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    result = llm._format_tools_prompt(tools)
    assert "get_weather" in result
    assert "Get the current weather" in result
    assert "tool_calls" in result
    assert "valid JSON" in result


def test_init_with_dict_config():
    llm = CodexLLM({"model": "gpt-5", "codex_cli_path": "/usr/bin/codex"})
    assert llm.config.model == "gpt-5"
    assert llm.config.codex_cli_path == "/usr/bin/codex"


def test_init_with_base_config():
    from mem0.configs.llms.base import BaseLlmConfig

    base_config = BaseLlmConfig(model="gpt-5", temperature=0.5)
    llm = CodexLLM(base_config)
    assert llm.config.model == "gpt-5"
    assert llm.config.temperature == 0.5
    assert isinstance(llm.config, CodexConfig)


def test_init_with_none_config():
    llm = CodexLLM()
    assert llm.config.model == "o4-mini"
    assert isinstance(llm.config, CodexConfig)
