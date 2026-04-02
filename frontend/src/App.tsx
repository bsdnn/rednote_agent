import { useEffect, useRef } from "react";
import "./index.css";
import NavBar from "./components/NavBar";
import InputForm from "./components/InputForm";
import LoadingState from "./components/LoadingState";
import EmptyState from "./components/EmptyState";
import ResultCard from "./components/ResultCard";
import CopyButton from "./components/CopyButton";
import RefinementBar from "./components/RefinementBar";
import { useSSEGenerate } from "./hooks/useSSEGenerate";
import { useConversationHistory } from "./hooks/useConversationHistory";
import type { GenerateRequest, UserPersona } from "./types/api";

export default function App() {
  const { phase, result, agentPlan, thinkingSteps, tokenUsage, error, generate, reset } = useSSEGenerate();
  const { history, addUserMessage, addAssistantResult, reset: resetHistory } = useConversationHistory();

  const lastPersonaRef = useRef<UserPersona>({
    skin_type: "combination",
    age_group: "25-30",
    preferences: [],
    budget: "mid-range",
  });
  const lastToneRef = useRef("活泼甜美");

  const isLoading = phase === "connecting" || phase === "tool_calling";

  useEffect(() => {
    if (phase === "complete" && result) {
      addAssistantResult(result);
    }
  }, [phase, result]);

  const handleGenerate = (query: string, tone: string, persona: UserPersona) => {
    lastPersonaRef.current = persona;
    lastToneRef.current = tone;
    resetHistory();
    addUserMessage(query);
    generate({ query, tone, max_iterations: 5, persona } as GenerateRequest);
  };

  const handleRefine = (request: object, endpoint: string) => {
    generate(request as GenerateRequest, endpoint);
  };

  const handleReset = () => {
    reset();
    resetHistory();
  };

  const showEmpty = phase === "idle";
  const showLoading = isLoading;
  const showResult = phase === "complete" && result;
  const showError = phase === "error" && error;

  return (
    <>
      <NavBar />
      <div className="app-layout">
        {/* Left panel — sticky form */}
        <aside className="left-panel">
          <div className="left-panel-heading">
            <div className="display">
              爆款<br />
              <span>文案</span>
            </div>
            <p>输入产品需求，AI Agent 实时生成小红书文案</p>
          </div>
          <InputForm onSubmit={handleGenerate} loading={isLoading} />
        </aside>

        {/* Right panel — result area */}
        <main className="right-panel">
          {showEmpty && <EmptyState />}

          {showLoading && <LoadingState steps={thinkingSteps} plan={agentPlan} />}

          {showError && (
            <div className="card error-card">
              <p className="error-text">⚠ {error}</p>
              <button className="retry-btn" onClick={handleReset} type="button">
                重试
              </button>
            </div>
          )}

          {showResult && (
            <>
              <ResultCard result={result} tokenUsage={tokenUsage} />
              <CopyButton result={result} />
              <RefinementBar
                previousResult={result}
                conversationHistory={history}
                persona={lastPersonaRef.current}
                tone={lastToneRef.current}
                onRefine={handleRefine}
                disabled={isLoading}
              />
            </>
          )}
        </main>
      </div>
    </>
  );
}
