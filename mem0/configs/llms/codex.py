from typing import Optional

from mem0.configs.llms.base import BaseLlmConfig


class CodexConfig(BaseLlmConfig):
    """
    Configuration class for OpenAI Codex SDK parameters.
    Inherits from BaseLlmConfig and adds Codex-specific settings.

    The Codex SDK uses ChatGPT subscription authentication (not API keys).
    It spawns the Codex CLI as a subprocess and communicates via JSON-RPC.
    """

    def __init__(
        self,
        # Base parameters
        model: Optional[str] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        reasoning_effort: Optional[str] = None,
        http_client_proxies: Optional[dict] = None,
        # Codex-specific parameters
        codex_cli_path: Optional[str] = None,
        full_context: bool = False,
    ):
        """
        Initialize Codex configuration.

        Args:
            model: Model to use via Codex, defaults to None (will use "o4-mini")
            temperature: Controls randomness, defaults to 0.1
            api_key: Not used by Codex (uses ChatGPT subscription), kept for interface compatibility
            max_tokens: Maximum tokens to generate, defaults to 2000
            top_p: Nucleus sampling parameter, defaults to 0.1
            top_k: Top-k sampling parameter, defaults to 1
            enable_vision: Enable vision capabilities, defaults to False
            vision_details: Vision detail level, defaults to "auto"
            reasoning_effort: Effort level for reasoning models, defaults to None
            http_client_proxies: HTTP client proxy settings, defaults to None
            codex_cli_path: Optional path to the Codex CLI binary, defaults to None (auto-detect)
            full_context: Enable Codex full-context mode, defaults to False
        """
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            reasoning_effort=reasoning_effort,
            http_client_proxies=http_client_proxies,
        )

        self.codex_cli_path = codex_cli_path
        self.full_context = full_context
