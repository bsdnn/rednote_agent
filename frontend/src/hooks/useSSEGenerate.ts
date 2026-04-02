import { useReducer, useCallback, useRef } from "react";
import type {
  AgentPlan,
  GeneratePhase,
  GenerateRequest,
  GenerateResponse,
  ThinkingStep,
  TokenUsage,
} from "../types/api";

interface State {
  phase: GeneratePhase;
  result: GenerateResponse | null;
  agentPlan: AgentPlan | null;
  thinkingSteps: ThinkingStep[];
  tokenUsage: TokenUsage | null;
  error: string | null;
}

type Action =
  | { type: "START" }
  | { type: "AGENT_PLAN"; plan: AgentPlan }
  | { type: "THINKING"; step: string; tool?: string }
  | { type: "TOOL_RESULT"; tool: string }
  | { type: "COMPLETE"; result: GenerateResponse }
  | { type: "TOKEN_USAGE"; usage: TokenUsage }
  | { type: "ERROR"; error: string }
  | { type: "RESET" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "START":
      return { phase: "connecting", result: null, agentPlan: null, thinkingSteps: [], tokenUsage: null, error: null };
    case "AGENT_PLAN":
      return { ...state, agentPlan: action.plan };
    case "THINKING":
      return {
        ...state,
        phase: "tool_calling",
        thinkingSteps: [
          ...state.thinkingSteps,
          { step: action.step, tool: action.tool, timestamp: Date.now() },
        ],
      };
    case "TOOL_RESULT":
      return {
        ...state,
        thinkingSteps: state.thinkingSteps.map((s, i) =>
          i === state.thinkingSteps.length - 1 ? { ...s, isToolResult: true } : s
        ),
      };
    case "COMPLETE":
      return { ...state, phase: "complete", result: action.result };
    case "TOKEN_USAGE":
      return { ...state, tokenUsage: action.usage };
    case "ERROR":
      return { ...state, phase: "error", error: action.error };
    case "RESET":
      return { phase: "idle", result: null, agentPlan: null, thinkingSteps: [], tokenUsage: null, error: null };
    default:
      return state;
  }
}

const initialState: State = {
  phase: "idle",
  result: null,
  agentPlan: null,
  thinkingSteps: [],
  tokenUsage: null,
  error: null,
};

export function useSSEGenerate() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const abortRef = useRef<AbortController | null>(null);

  const generate = useCallback(
    async (request: GenerateRequest, endpoint = "/api/generate") => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      dispatch({ type: "START" });

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request),
          signal: controller.signal,
        });

        if (!response.ok) {
          const text = await response.text();
          throw new Error(`HTTP ${response.status}: ${text}`);
        }

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let currentEvent = "";
        let currentData = "";

        const dispatchEvent = (eventType: string, rawData: string) => {
          try {
            const data = JSON.parse(rawData);
            switch (eventType) {
              case "agent_plan":
                dispatch({ type: "AGENT_PLAN", plan: data as AgentPlan });
                break;
              case "agent_thinking":
                dispatch({ type: "THINKING", step: data.step, tool: data.tool });
                break;
              case "tool_result":
                dispatch({ type: "TOOL_RESULT", tool: data.tool });
                break;
              case "complete":
                dispatch({ type: "COMPLETE", result: data as GenerateResponse });
                break;
              case "token_usage":
                dispatch({ type: "TOKEN_USAGE", usage: data as TokenUsage });
                break;
              case "error":
                dispatch({ type: "ERROR", error: data.message });
                break;
            }
          } catch {
            // ignore malformed events
          }
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (line.startsWith("event:")) {
              currentEvent = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              currentData = line.slice(5).trim();
            } else if (line.trim() === "") {
              if (currentEvent && currentData) {
                dispatchEvent(currentEvent, currentData);
              }
              currentEvent = "";
              currentData = "";
            }
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          dispatch({ type: "ERROR", error: (err as Error).message });
        }
      }
    },
    []
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    dispatch({ type: "RESET" });
  }, []);

  return { ...state, generate, reset };
}
