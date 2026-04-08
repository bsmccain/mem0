import OpenAI from "openai";
import { spawn } from "node:child_process";
import { LLM, LLMResponse } from "./base";
import { LLMConfig, Message } from "../types";

export class OpenAILLM implements LLM {
  private openai: OpenAI | null;
  private model: string;
  private useCodexCLI: boolean;
  private codexCLICommand?: string[];

  constructor(config: LLMConfig) {
    this.useCodexCLI =
      Boolean(config.useCodexCLI) || process.env.MEM0_USE_CODEX_CLI === "1";
    this.codexCLICommand = config.codexCLICommand;

    if (this.useCodexCLI) {
      if (!this.codexCLICommand?.length) {
        throw new Error(
          "useCodexCLI=true requires codexCLICommand (e.g. ['codex', 'exec', '--json'])",
        );
      }
      this.openai = null;
    } else {
      this.openai = new OpenAI({
        apiKey: config.apiKey,
        baseURL: config.baseURL,
      });
    }

    this.model = config.model || "gpt-4.1-nano-2025-04-14";
  }

  private toChatMessages(messages: Message[]) {
    return messages.map((msg) => {
      const role = msg.role as "system" | "user" | "assistant";
      return {
        role,
        content:
          typeof msg.content === "string"
            ? msg.content
            : JSON.stringify(msg.content),
      };
    });
  }

  private async runCodexCLI(payload: Record<string, unknown>) {
    const [command, ...args] = this.codexCLICommand!;

    return new Promise<any>((resolve, reject) => {
      const child = spawn(command, args, { stdio: ["pipe", "pipe", "pipe"] });
      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });

      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });

      child.on("error", (err) => {
        reject(err);
      });

      child.on("close", (code) => {
        if (code !== 0) {
          reject(
            new Error(
              `Codex CLI command failed with code ${code}: ${stderr.trim() || stdout.trim()}`,
            ),
          );
          return;
        }

        const output = stdout.trim();
        if (!output) {
          reject(new Error("Codex CLI command returned empty output"));
          return;
        }

        try {
          resolve(JSON.parse(output));
        } catch {
          reject(new Error("Codex CLI command returned invalid JSON output"));
        }
      });

      child.stdin.write(JSON.stringify(payload));
      child.stdin.end();
    });
  }

  async generateResponse(
    messages: Message[],
    responseFormat?: { type: string },
    tools?: any[],
  ): Promise<string | LLMResponse> {
    if (this.useCodexCLI) {
      const parsed = await this.runCodexCLI({
        provider: "openai",
        params: {
          messages: this.toChatMessages(messages),
          model: this.model,
          response_format: responseFormat,
          ...(tools && { tools, tool_choice: "auto" }),
        },
      });

      if (parsed?.choices?.[0]?.message) {
        const response = parsed.choices[0].message;
        if (response.tool_calls) {
          return {
            content: response.content || "",
            role: response.role || "assistant",
            toolCalls: response.tool_calls.map((call: any) => ({
              name: call.function?.name || call.name,
              arguments: call.function?.arguments || call.arguments,
            })),
          };
        }
        return response.content || "";
      }

      if (parsed?.content !== undefined) {
        if (tools) {
          return {
            content: parsed.content || "",
            role: "assistant",
            toolCalls: (parsed.tool_calls || []).map((call: any) => ({
              name: call.function?.name || call.name,
              arguments: call.function?.arguments || call.arguments,
            })),
          };
        }
        return parsed.content || "";
      }

      throw new Error(
        "Unsupported Codex CLI output format. Expected OpenAI-like {'choices': ...} or {'content': ...}.",
      );
    }

    const completion = await this.openai!.chat.completions.create({
      messages: this.toChatMessages(messages),
      model: this.model,
      response_format: responseFormat as { type: "text" | "json_object" },
      ...(tools && { tools, tool_choice: "auto" }),
    });

    const response = completion.choices[0].message;

    if (response.tool_calls) {
      return {
        content: response.content || "",
        role: response.role,
        toolCalls: response.tool_calls.map((call) => ({
          name: call.function.name,
          arguments: call.function.arguments,
        })),
      };
    }

    return response.content || "";
  }

  async generateChat(messages: Message[]): Promise<LLMResponse> {
    if (this.useCodexCLI) {
      const parsed = await this.runCodexCLI({
        provider: "openai",
        params: {
          messages: this.toChatMessages(messages),
          model: this.model,
        },
      });

      if (parsed?.choices?.[0]?.message) {
        const response = parsed.choices[0].message;
        return {
          content: response.content || "",
          role: response.role || "assistant",
        };
      }

      if (parsed?.content !== undefined) {
        return {
          content: parsed.content || "",
          role: "assistant",
        };
      }

      throw new Error(
        "Unsupported Codex CLI output format. Expected OpenAI-like {'choices': ...} or {'content': ...}.",
      );
    }

    const completion = await this.openai!.chat.completions.create({
      messages: this.toChatMessages(messages),
      model: this.model,
    });
    const response = completion.choices[0].message;
    return {
      content: response.content || "",
      role: response.role,
    };
  }
}
