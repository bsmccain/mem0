import { LLM, LLMResponse } from "./base";
import { LLMConfig, Message } from "../types";

/**
 * LLM provider that uses the OpenAI Codex SDK to access OpenAI models.
 *
 * The Codex SDK spawns the Codex CLI as a subprocess and communicates via JSON-RPC.
 * It uses ChatGPT subscription authentication rather than API keys.
 *
 * Requires the `@openai/codex-sdk` package: pnpm add @openai/codex-sdk
 */
export class CodexLLM implements LLM {
  private model: string;

  constructor(config: LLMConfig) {
    this.model = config.model || "o4-mini";
  }

  /**
   * Flatten a messages array into a single prompt string for Codex.
   */
  private formatMessages(messages: Message[]): string {
    return messages
      .map((msg) => {
        const content =
          typeof msg.content === "string"
            ? msg.content
            : JSON.stringify(msg.content);
        switch (msg.role) {
          case "system":
            return `System: ${content}`;
          case "assistant":
            return `Assistant: ${content}`;
          default:
            return `User: ${content}`;
        }
      })
      .join("\n\n");
  }

  /**
   * Convert tool definitions into prompt instructions for Codex.
   */
  private formatToolsPrompt(tools: any[]): string {
    const toolDescriptions = tools
      .filter((tool) => tool.type === "function")
      .map((tool) => {
        const func = tool.function;
        const name = func.name || "unknown";
        const description = func.description || "";
        const parameters = JSON.stringify(func.parameters || {});
        return `- ${name}: ${description}\n  Parameters: ${parameters}`;
      })
      .join("\n");

    return (
      `\n\nYou have access to the following tools:\n${toolDescriptions}\n\n` +
      "If you need to call a tool, respond with a JSON object in this exact format:\n" +
      '{"tool_calls": [{"name": "<function_name>", "arguments": {<arguments>}}], ' +
      '"content": "<optional message>"}\n' +
      "You must respond with valid JSON only."
    );
  }

  /**
   * Run a prompt through the Codex SDK and return the text response.
   */
  private async runCodex(prompt: string): Promise<string> {
    let Codex: any;
    try {
      const codexModule = await import("@openai/codex-sdk");
      Codex = codexModule.Codex || codexModule.default;
    } catch {
      throw new Error(
        "The @openai/codex-sdk package is required for the Codex LLM provider. " +
          "Install it with: pnpm add @openai/codex-sdk",
      );
    }

    const codex = new Codex();
    const thread = codex.startThread({ model: this.model });
    const turn = await thread.run(prompt);
    return turn.finalResponse || "";
  }

  async generateResponse(
    messages: Message[],
    responseFormat?: { type: string },
    tools?: any[],
  ): Promise<string | LLMResponse> {
    let prompt = this.formatMessages(messages);

    if (
      responseFormat &&
      responseFormat.type === "json_object" &&
      !tools
    ) {
      prompt += "\n\nYou must respond with valid JSON only.";
    }

    if (tools) {
      prompt += this.formatToolsPrompt(tools);
    }

    const responseText = await this.runCodex(prompt);

    if (tools) {
      try {
        const parsed = JSON.parse(responseText);
        const toolCalls = (parsed.tool_calls || []).map(
          (tc: { name: string; arguments: any }) => ({
            name: tc.name || "",
            arguments:
              typeof tc.arguments === "string"
                ? tc.arguments
                : JSON.stringify(tc.arguments || {}),
          }),
        );
        return {
          content: parsed.content || responseText,
          role: "assistant",
          toolCalls,
        };
      } catch {
        return {
          content: responseText,
          role: "assistant",
          toolCalls: [],
        };
      }
    }

    return responseText;
  }

  async generateChat(messages: Message[]): Promise<LLMResponse> {
    const prompt = this.formatMessages(messages);
    const responseText = await this.runCodex(prompt);
    return {
      content: responseText,
      role: "assistant",
    };
  }
}
