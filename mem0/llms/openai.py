import json
import logging
import os
import subprocess
from typing import Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json


class OpenAILLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, OpenAIConfig, Dict]] = None):
        # Convert to OpenAIConfig if needed
        if config is None:
            config = OpenAIConfig()
        elif isinstance(config, dict):
            config = OpenAIConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OpenAIConfig):
            # Convert BaseLlmConfig to OpenAIConfig
            config = OpenAIConfig(
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
            self.config.model = "gpt-4.1-nano-2025-04-14"

        self._use_codex_cli = bool(self.config.use_codex_cli or os.getenv("MEM0_USE_CODEX_CLI") == "1")

        if self._use_codex_cli:
            self.client = None
            self._codex_cli_command = self.config.codex_cli_command
            if not self._codex_cli_command:
                raise ValueError(
                    "use_codex_cli=True requires codex_cli_command (e.g. ['codex', 'exec', '--json'])."
                )
            return

        if os.environ.get("OPENROUTER_API_KEY"):  # Use OpenRouter
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
            )
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = self.config.openai_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _parse_response(self, response, tools):
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def _generate_response_with_codex_cli(self, params: Dict, tools: Optional[List[Dict]] = None):
        payload = {
            "provider": "openai",
            "params": params,
            "tools": tools,
        }

        completed = subprocess.run(
            self._codex_cli_command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Codex CLI command failed with code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )

        output = completed.stdout.strip()
        if not output:
            raise RuntimeError("Codex CLI command returned empty output")

        parsed = json.loads(output)

        if isinstance(parsed, dict) and "choices" in parsed:
            class _Msg:
                def __init__(self, msg):
                    self.content = msg.get("content")
                    self.tool_calls = msg.get("tool_calls")

            class _Choice:
                def __init__(self, choice):
                    self.message = _Msg(choice.get("message", {}))

            class _Response:
                def __init__(self, raw):
                    self.choices = [_Choice(raw["choices"][0])]

            return self._parse_response(_Response(parsed), tools)

        if isinstance(parsed, dict) and "content" in parsed:
            if tools:
                return {
                    "content": parsed.get("content"),
                    "tool_calls": parsed.get("tool_calls", []),
                }
            return parsed.get("content")

        raise RuntimeError(
            "Unsupported Codex CLI output format. Expected OpenAI-like {'choices': ...} or {'content': ...}."
        )

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        params = self._get_supported_params(messages=messages, **kwargs)

        params.update(
            {
                "model": self.config.model,
                "messages": messages,
            }
        )

        if os.getenv("OPENROUTER_API_KEY") and not self._use_codex_cli:
            openrouter_params = {}
            if self.config.models:
                openrouter_params["models"] = self.config.models
                openrouter_params["route"] = self.config.route
                params.pop("model")

            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)

        elif not self._use_codex_cli:
            openai_specific_generation_params = ["store"]
            for param in openai_specific_generation_params:
                if hasattr(self.config, param):
                    params[param] = getattr(self.config, param)

        if response_format:
            params["response_format"] = response_format
        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        if self._use_codex_cli:
            parsed_response = self._generate_response_with_codex_cli(params=params, tools=tools)
            response = {"codex_cli": True, "raw_output": parsed_response}
        else:
            response = self.client.chat.completions.create(**params)
            parsed_response = self._parse_response(response, tools)

        if self.config.response_callback:
            try:
                self.config.response_callback(self, response, params)
            except Exception as e:
                logging.error(f"Error due to callback: {e}")
                pass
        return parsed_response
