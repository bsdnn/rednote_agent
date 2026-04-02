import { useState } from "react";
import type { GenerateResponse } from "../types/api";

interface CopyButtonProps {
  result: GenerateResponse;
}

export default function CopyButton({ result }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const text = [
      result.title,
      "",
      result.body,
      "",
      result.hashtags.join(" "),
      result.emojis.join(" "),
    ]
      .filter(Boolean)
      .join("\n");

    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const el = document.createElement("textarea");
      el.value = text;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
    }

    setCopied(true);
    setTimeout(() => setCopied(false), 2200);
  };

  return (
    <button
      className={`copy-btn ${copied ? "copied" : ""}`}
      onClick={handleCopy}
      type="button"
    >
      {copied ? "✓ 已复制到剪贴板" : "复制全文"}
    </button>
  );
}
