import { useState } from "react";
import type { UserPersona } from "../types/api";

const SKIN_TYPES = [
  { value: "oily", label: "油皮" },
  { value: "dry", label: "干皮" },
  { value: "combination", label: "混合皮" },
  { value: "sensitive", label: "敏感皮" },
  { value: "normal", label: "中性皮" },
] as const;

const AGE_GROUPS = ["18-24", "25-30", "31-40", "41+"] as const;

const PREFERENCE_OPTIONS = [
  "美白提亮", "补水保湿", "抗氧化", "抗老紧致",
  "去痘控油", "修护敏感", "防晒", "眼部护理",
];

const BUDGETS = [
  { value: "budget", label: "平价党" },
  { value: "mid-range", label: "中端" },
  { value: "luxury", label: "轻奢" },
] as const;

interface PersonaPanelProps {
  value: UserPersona;
  onChange: (p: UserPersona) => void;
}

export default function PersonaPanel({ value, onChange }: PersonaPanelProps) {
  const [open, setOpen] = useState(false);

  const update = (patch: Partial<UserPersona>) => onChange({ ...value, ...patch });

  const togglePref = (pref: string) => {
    const prefs = value.preferences.includes(pref)
      ? value.preferences.filter((p) => p !== pref)
      : [...value.preferences, pref];
    update({ preferences: prefs });
  };

  return (
    <>
      <div className="section-divider" />
      <button
        className={`persona-toggle ${open ? "open" : ""}`}
        onClick={() => setOpen(!open)}
        type="button"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
          <circle cx="7" cy="4" r="2.5" />
          <path d="M1 12c0-3.3 2.7-5 6-5s6 1.7 6 5H1z" />
        </svg>
        用户画像（可选）
        <svg
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <path d="M2 4l4 4 4-4" />
        </svg>
      </button>

      <div className={`persona-panel-body ${open ? "open" : ""}`}>
        <div className="persona-panel-inner">
          <div className="persona-content">
            <div className="persona-row">
              <span className="persona-row-label">肤质</span>
              <div className="persona-chips">
                {SKIN_TYPES.map(({ value: v, label }) => (
                  <button
                    key={v}
                    type="button"
                    className={`persona-chip ${value.skin_type === v ? "selected" : ""}`}
                    onClick={() => update({ skin_type: v })}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className="persona-row">
              <span className="persona-row-label">年龄段</span>
              <div className="persona-chips">
                {AGE_GROUPS.map((age) => (
                  <button
                    key={age}
                    type="button"
                    className={`persona-chip ${value.age_group === age ? "selected" : ""}`}
                    onClick={() => update({ age_group: age })}
                  >
                    {age}
                  </button>
                ))}
              </div>
            </div>

            <div className="persona-row">
              <span className="persona-row-label">护肤关注</span>
              <div className="persona-chips">
                {PREFERENCE_OPTIONS.map((pref) => (
                  <button
                    key={pref}
                    type="button"
                    className={`persona-chip ${value.preferences.includes(pref) ? "selected" : ""}`}
                    onClick={() => togglePref(pref)}
                  >
                    {pref}
                  </button>
                ))}
              </div>
            </div>

            <div className="persona-row">
              <span className="persona-row-label">预算</span>
              <div className="persona-chips">
                {BUDGETS.map(({ value: v, label }) => (
                  <button
                    key={v}
                    type="button"
                    className={`persona-chip ${value.budget === v ? "selected" : ""}`}
                    onClick={() => update({ budget: v })}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
