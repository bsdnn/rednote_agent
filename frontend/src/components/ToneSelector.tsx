const TONES = ["活泼甜美", "专业种草", "温柔治愈", "幽默搞笑", "精英范儿"];

interface ToneSelectorProps {
  value: string;
  onChange: (tone: string) => void;
}

export default function ToneSelector({ value, onChange }: ToneSelectorProps) {
  return (
    <>
      <div className="section-divider" />
      <label className="input-label">文案风格</label>
      <div className="tone-grid">
        {TONES.map((tone) => (
          <button
            key={tone}
            className={`tone-pill ${value === tone ? "active" : ""}`}
            onClick={() => onChange(tone)}
            type="button"
          >
            {tone}
          </button>
        ))}
      </div>
    </>
  );
}
