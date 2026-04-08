import { EventEmitter } from "events";
import { OpenAILLM } from "../src/llms/openai";

jest.mock("openai", () => {
  return jest.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: jest.fn(),
      },
    },
  }));
});

jest.mock("node:child_process", () => ({
  spawn: jest.fn(),
}));

import { spawn } from "node:child_process";

function mockSpawn({ code, stdout = "", stderr = "" }: { code: number; stdout?: string; stderr?: string }) {
  const child = new EventEmitter() as any;
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.stdin = {
    write: jest.fn(),
    end: jest.fn(() => {
      if (stdout) child.stdout.emit("data", Buffer.from(stdout));
      if (stderr) child.stderr.emit("data", Buffer.from(stderr));
      child.emit("close", code);
    }),
  };
  (spawn as jest.Mock).mockReturnValue(child);
}

describe("OpenAILLM Codex CLI mode", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("requires codexCLICommand when useCodexCLI is enabled", () => {
    expect(() => new OpenAILLM({ useCodexCLI: true })).toThrow(
      "useCodexCLI=true requires codexCLICommand",
    );
  });

  test("returns content from simplified Codex output", async () => {
    mockSpawn({ code: 0, stdout: '{"content":"from codex"}' });

    const llm = new OpenAILLM({
      useCodexCLI: true,
      codexCLICommand: ["codex", "exec", "--json"],
    });

    const result = await llm.generateResponse([{ role: "user", content: "hello" }]);
    expect(result).toBe("from codex");
    expect(spawn).toHaveBeenCalledWith("codex", ["exec", "--json"], {
      stdio: ["pipe", "pipe", "pipe"],
    });
  });

  test("throws when Codex CLI exits non-zero", async () => {
    mockSpawn({ code: 1, stderr: "boom" });

    const llm = new OpenAILLM({
      useCodexCLI: true,
      codexCLICommand: ["codex", "exec", "--json"],
    });

    await expect(
      llm.generateResponse([{ role: "user", content: "hello" }]),
    ).rejects.toThrow("Codex CLI command failed");
  });
});
