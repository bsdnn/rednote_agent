import { useEffect, useState } from "react";

export default function NavBar() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.ok ? setOnline(true) : setOnline(false))
      .catch(() => setOnline(false));
  }, []);

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        小红书<span>·</span>文案生成器
      </div>
      <div className="health-indicator">
        <span className={`health-dot${online === false ? " offline" : ""}`} />
        {online === null ? "检测中..." : online ? "AI 在线" : "API 离线"}
      </div>
    </nav>
  );
}
