/// <reference types="jest" />
/**
 * Codex LLM — unit tests (mocked @openai/codex-sdk).
 */

import { CodexLLM } from "../src/llms/codex";

const mockRun = jest.fn();
const mockStartThread = jest.fn().mockReturnValue({ run: mockRun });

jest.mock(
  "@openai/codex-sdk",
  () => ({
    Codex: jest.fn().mockImplementation(() => ({
      startThread: mockStartThread,
    })),
  }),
  { virtual: true },
);

describe("CodexLLM (unit)", () => {
  beforeEach(() => {
    mockRun.mockClear();
    mockStartThread.mockClear();
    mockStartThread.mockReturnValue({ run: mockRun });
  });

  it("generateResponse() returns a text response", async () => {
    mockRun.mockResolvedValueOnce({
      finalResponse: "Hello, world!",
    });

    const llm = new CodexLLM({ model: "o4-mini" });
    const result = await llm.generateResponse([
      { role: "user", content: "Hi" },
    ]);

    expect(mockRun).toHaveBeenCalledTimes(1);
    expect(result).toBe("Hello, world!");
  });

  it("generateResponse() appends JSON instruction for json_object format", async () => {
    mockRun.mockResolvedValueOnce({
      finalResponse: '{"facts": ["sky is blue"]}',
    });

    const llm = new CodexLLM({ model: "o4-mini" });
    const result = await llm.generateResponse(
      [{ role: "user", content: "Extract facts" }],
      { type: "json_object" },
    );

    const prompt = mockRun.mock.calls[0][0];
    expect(prompt).toContain("You must respond with valid JSON only.");
    expect(result).toBe('{"facts": ["sky is blue"]}');
  });

  it("generateResponse() handles tool calls", async () => {
    mockRun.mockResolvedValueOnce({
      finalResponse: JSON.stringify({
        tool_calls: [
          { name: "get_weather", arguments: { city: "London" } },
        ],
        content: "Here's the weather.",
      }),
    });

    const llm = new CodexLLM({ model: "o4-mini" });
    const result = await llm.generateResponse(
      [{ role: "user", content: "What is the weather?" }],
      undefined,
      [{ type: "function", function: { name: "get_weather" } }],
    );

    expect(result).toEqual({
      content: "Here's the weather.",
      role: "assistant",
      toolCalls: [
        { name: "get_weather", arguments: '{"city":"London"}' },
      ],
    });
  });

  it("generateResponse() returns empty tool_calls on parse failure", async () => {
    mockRun.mockResolvedValueOnce({
      finalResponse: "This is not valid JSON",
    });

    const llm = new CodexLLM({ model: "o4-mini" });
    const result = await llm.generateResponse(
      [{ role: "user", content: "Do something" }],
      undefined,
      [{ type: "function", function: { name: "test_tool" } }],
    );

    expect(result).toEqual({
      content: "This is not valid JSON",
      role: "assistant",
      toolCalls: [],
    });
  });

  it("generateChat() returns LLMResponse shape", async () => {
    mockRun.mockResolvedValueOnce({
      finalResponse: "I can help with that.",
    });

    const llm = new CodexLLM({ model: "o4-mini" });
    const result = await llm.generateChat([
      { role: "user", content: "Help me" },
    ]);

    expect(result).toEqual({
      content: "I can help with that.",
      role: "assistant",
    });
  });

  it("formats system and user messages correctly", async () => {
    mockRun.mockResolvedValueOnce({
      finalResponse: "Response",
    });

    const llm = new CodexLLM({ model: "o4-mini" });
    await llm.generateResponse([
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hello" },
    ]);

    const prompt = mockRun.mock.calls[0][0];
    expect(prompt).toContain("System: You are helpful.");
    expect(prompt).toContain("User: Hello");
  });

  it("uses default model when none specified", async () => {
    const llm = new CodexLLM({});
    expect((llm as any).model).toBe("o4-mini");
  });
});
