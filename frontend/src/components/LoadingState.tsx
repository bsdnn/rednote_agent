import type { AgentPlan, ThinkingStep } from "../types/api";
import Card from "./Card";

interface LoadingStateProps {
  steps: ThinkingStep[];
  plan?: AgentPlan | null;
}

export default function LoadingState({ steps, plan }: LoadingStateProps) {
  return (
    <Card className="terminal-card">
      <div className="terminal-header">Generating</div>

      {plan && (
        <div className="agent-plan">
          <div className="agent-plan-goal">🎯 {plan.goal}</div>
          <div className="agent-plan-steps">
            {plan.steps.map((s, i) => (
              <span key={i} className="agent-plan-step">
                {i + 1}. {s}
              </span>
            ))}
          </div>
          {plan.key_focus && (
            <div className="agent-plan-focus">核心切入点：{plan.key_focus}</div>
          )}
        </div>
      )}

      <div className="terminal-lines">
        {steps.map((s, i) => {
          const isLast = i === steps.length - 1;
          const isDone = !isLast || s.isToolResult;
          const isTool = Boolean(s.tool) && !s.isToolResult;

          let lineClass = "terminal-line";
          if (isDone) lineClass += " done";
          else if (isTool) lineClass += " tool active";
          else lineClass += " active";

          return (
            <div key={s.timestamp} className={lineClass}>
              <i className="terminal-icon">
                {isDone ? "✓" : isTool ? "→" : "●"}
              </i>
              <span>
                {s.step}
                {!isDone && <span className="terminal-cursor" />}
              </span>
            </div>
          );
        })}
        {steps.length === 0 && (
          <div className="terminal-line active">
            <i className="terminal-icon">●</i>
            <span>连接中<span className="terminal-cursor" /></span>
          </div>
        )}
      </div>
    </Card>
  );
}
