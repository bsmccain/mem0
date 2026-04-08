import asyncio
import json
import logging
from typing import Dict, List, Optional, Union

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.codex import CodexConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json

logger = logging.getLogger(__name__)


class CodexLLM(LLMBase):
    """
    LLM provider that uses the OpenAI Codex SDK to access OpenAI models.

    The Codex SDK spawns the Codex CLI as a subprocess and communicates via JSON-RPC.
    It uses ChatGPT subscription authentication rather than API keys.

    Requires the `openai-codex-sdk` package: pip install openai-codex-sdk
    """

    def __init__(self, config: Optional[Union[BaseLlmConfig, CodexConfig, Dict]] = None):
        if config is None:
            config = CodexConfig()
        elif isinstance(config, dict):
            config = CodexConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, CodexConfig):
            config = CodexConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                reasoning_effort=getattr(config, "reasoning_effort", None),
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "o4-mini"

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Flatten a messages array into a single prompt string for Codex.

        System messages are prefixed with "System:", user messages with "User:",
        and assistant messages with "Assistant:".
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        return "\n\n".join(parts)

    def _format_tools_prompt(self, tools: List[Dict]) -> str:
        """
        Convert tool definitions into prompt instructions for Codex.

        Since Codex doesn't have native function calling, we describe the available
        tools in the prompt and instruct the model to respond with a JSON object
        containing tool calls.
        """
        tool_descriptions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                name = func.get("name", "unknown")
                description = func.get("description", "")
                parameters = json.dumps(func.get("parameters", {}))
                tool_descriptions.append(f"- {name}: {description}\n  Parameters: {parameters}")

        tools_text = "\n".join(tool_descriptions)
        return (
            f"\n\nYou have access to the following tools:\n{tools_text}\n\n"
            "If you need to call a tool, respond with a JSON object in this exact format:\n"
            '{"tool_calls": [{"name": "<function_name>", "arguments": {<arguments>}}], '
            '"content": "<optional message>"}\n'
            "You must respond with valid JSON only."
        )

    def _run_codex(self, prompt: str) -> str:
        """
        Run a prompt through the Codex SDK and return the text response.
        """
        try:
            from openai_codex_sdk import Codex
        except ImportError:
            raise ImportError(
                "The openai-codex-sdk package is required for the Codex LLM provider. "
                "Install it with: pip install openai-codex-sdk"
            )

        async def _async_run():
            codex = Codex()
            thread = codex.start_thread(model=self.config.model)
            turn = await thread.run(prompt)
            return turn.final_response

        return asyncio.run(_async_run())

    def _parse_response(self, response_text: str, tools: Optional[List[Dict]]) -> Union[str, Dict]:
        """
        Parse the Codex response text.

        If tools were provided, attempt to parse the response as JSON containing tool calls.
        Otherwise, return the raw text.
        """
        if tools:
            try:
                json_str = extract_json(response_text)
                parsed = json.loads(json_str)
                tool_calls = []
                for tc in parsed.get("tool_calls", []):
                    arguments = tc.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    tool_calls.append({"name": tc.get("name", ""), "arguments": arguments})
                return {
                    "content": parsed.get("content", response_text),
                    "tool_calls": tool_calls,
                }
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse tool calls from Codex response: {e}")
                return {"content": response_text, "tool_calls": []}
        return response_text

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response using the OpenAI Codex SDK.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (dict, optional): Format of the response. Supports {"type": "json_object"}.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional parameters (unused by Codex).

        Returns:
            str or dict: The generated response.
        """
        prompt = self._format_messages(messages)

        if response_format and response_format.get("type") == "json_object" and not tools:
            prompt += "\n\nYou must respond with valid JSON only."

        if tools:
            prompt += self._format_tools_prompt(tools)

        response_text = self._run_codex(prompt)
        return self._parse_response(response_text, tools)
