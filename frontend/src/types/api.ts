export interface UserPersona {
  skin_type: "oily" | "dry" | "combination" | "sensitive" | "normal";
  age_group: "18-24" | "25-30" | "31-40" | "41+";
  preferences: string[];
  budget: "budget" | "mid-range" | "luxury";
}

export interface AgentPlan {
  goal: string;
  steps: string[];
  key_focus: string;
}

export interface GenerateRequest {
  query: string;
  tone: string;
  max_iterations?: number;
  persona?: UserPersona;
  user_id?: string;
}

export interface GenerateResponse {
  title: string;
  body: string;
  hashtags: string[];
  emojis: string[];
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export type SSEEventType =
  | "agent_thinking"
  | "tool_result"
  | "complete"
  | "token_usage"
  | "error";

export type GeneratePhase =
  | "idle"
  | "connecting"
  | "tool_calling"
  | "complete"
  | "error";

export interface ThinkingStep {
  step: string;
  tool?: string;
  timestamp: number;
  isToolResult?: boolean;
}
