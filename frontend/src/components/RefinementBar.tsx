import { useState } from "react";
import type { GenerateResponse, UserPersona } from "../types/api";
import Card from "./Card";

interface RefinementBarProps {
  previousResult: GenerateResponse;
  conversationHistory: Array<{ role: string; content: string }>;
  persona: UserPersona;
  tone: string;
  onRefine: (request: object, endpoint: string) => void;
  disabled: boolean;
}

export default function RefinementBar({
  previousResult,
  conversationHistory,
  persona,
  tone,
  onRefine,
  disabled,
}: RefinementBarProps) {
  const [instruction, setInstruction] = useState("");

  const handleSubmit = () => {
    const trimmed = instruction.trim();
    if (!trimmed) return;
    onRefine(
      {
        previous_result: previousResult,
        refinement_instruction: trimmed,
        conversation_history: conversationHistory,
        persona,
        tone,
      },
      "/api/refine"
    );
    setInstruction("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <Card className="refine-card">
      <div className="refine-label">继续润色</div>
      <div className="refine-row">
        <input
          className="refine-input"
          placeholder="例如：改得更专业一点、增加更多话题标签..."
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          className="refine-btn"
          onClick={handleSubmit}
          disabled={disabled || !instruction.trim()}
          type="button"
        >
          润色
        </button>
      </div>
    </Card>
  );
}
