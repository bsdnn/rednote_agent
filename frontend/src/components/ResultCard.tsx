import type { GenerateResponse, TokenUsage } from "../types/api";
import { useTypewriter } from "../hooks/useTypewriter";
import Card from "./Card";

interface ResultCardProps {
  result: GenerateResponse;
  tokenUsage?: TokenUsage | null;
}

export default function ResultCard({ result, tokenUsage }: ResultCardProps) {
  const displayedTitle = useTypewriter(result.title, 40);
  const displayedBody = useTypewriter(result.body, 18);

  const titleDone = displayedTitle.length >= result.title.length;
  const bodyDone = displayedBody.length >= result.body.length;

  return (
    <Card className="result-card">
      <div className="result-section">
        <div className="result-section-label">标题</div>
        <div className="result-title">
          {displayedTitle}
          {!titleDone && <span className="cursor-blink" />}
        </div>
      </div>

      <div className="result-divider" />

      <div className="result-section">
        <div className="result-section-label">正文</div>
        <div className="result-body">
          {displayedBody}
          {titleDone && !bodyDone && <span className="cursor-blink" />}
        </div>
      </div>

      {result.hashtags.length > 0 && (
        <>
          <div className="result-divider" />
          <div className="result-section">
            <div className="result-section-label">话题标签</div>
            <div className="hashtag-list">
              {result.hashtags.map((tag, i) => (
                <span
                  key={i}
                  className="hashtag-chip"
                  style={{ animationDelay: `${i * 60}ms` }}
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </>
      )}

      {result.emojis.length > 0 && (
        <>
          <div className="result-divider" />
          <div className="result-section">
            <div className="result-section-label">表情</div>
            <div className="emoji-list">
              {result.emojis.map((e, i) => (
                <span key={i}>{e}</span>
              ))}
            </div>
          </div>
        </>
      )}

      {tokenUsage && (
        <div className="token-badge">
          消耗 {tokenUsage.total_tokens.toLocaleString()} tokens
          &nbsp;·&nbsp;
          {tokenUsage.prompt_tokens} prompt + {tokenUsage.completion_tokens} completion
        </div>
      )}
    </Card>
  );
}
