import { useState } from "react";
import type { UserPersona } from "../types/api";
import ToneSelector from "./ToneSelector";
import PersonaPanel from "./PersonaPanel";
import Card from "./Card";

const DEFAULT_PERSONA: UserPersona = {
  skin_type: "combination",
  age_group: "25-30",
  preferences: [],
  budget: "mid-range",
};

interface InputFormProps {
  onSubmit: (query: string, tone: string, persona: UserPersona) => void;
  loading: boolean;
}

export default function InputForm({ onSubmit, loading }: InputFormProps) {
  const [query, setQuery] = useState("");
  const [tone, setTone] = useState("活泼甜美");
  const [persona, setPersona] = useState<UserPersona>(DEFAULT_PERSONA);
  const [error, setError] = useState("");

  const handleSubmit = () => {
    const trimmed = query.trim();
    if (!trimmed) {
      setError("请输入产品或需求关键词");
      return;
    }
    setError("");
    onSubmit(trimmed, tone, persona);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  };

  return (
    <Card className="input-card">
      <label className="input-label">输入产品或需求</label>
      <textarea
        className={`input-textarea ${error ? "error" : ""}`}
        placeholder="例如：去痘印、补水保湿面霜、卸妆油推荐、熬夜后急救..."
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          if (e.target.value.trim()) setError("");
        }}
        onKeyDown={handleKeyDown}
        disabled={loading}
        rows={3}
      />
      {error && <div className="input-error">{error}</div>}

      <ToneSelector value={tone} onChange={setTone} />
      <PersonaPanel value={persona} onChange={setPersona} />

      <button
        className="generate-btn"
        onClick={handleSubmit}
        disabled={loading}
        type="button"
      >
        {loading ? "AI 生成中..." : "生成爆款文案"}
      </button>
    </Card>
  );
}
