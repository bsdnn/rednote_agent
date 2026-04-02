import { useState, useCallback } from "react";
import type { GenerateResponse } from "../types/api";

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

export function useConversationHistory() {
  const [history, setHistory] = useState<ConversationMessage[]>([]);

  const addUserMessage = useCallback((content: string) => {
    setHistory((prev) => [...prev, { role: "user", content }]);
  }, []);

  const addAssistantResult = useCallback((result: GenerateResponse) => {
    setHistory((prev) => [
      ...prev,
      { role: "assistant", content: JSON.stringify(result) },
    ]);
  }, []);

  const reset = useCallback(() => setHistory([]), []);

  return { history, addUserMessage, addAssistantResult, reset };
}
