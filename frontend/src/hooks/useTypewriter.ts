import { useState, useEffect, useRef } from "react";

export function useTypewriter(text: string, baseSpeed = 25): string {
  const [displayed, setDisplayed] = useState("");
  const indexRef = useRef(0);
  const prevTextRef = useRef("");

  useEffect(() => {
    if (text !== prevTextRef.current) {
      prevTextRef.current = text;
      indexRef.current = 0;
      setDisplayed("");
    }
    if (indexRef.current >= text.length) return;

    // Faster for longer text
    const speed = Math.max(8, baseSpeed - Math.floor(text.length / 80));

    const timer = setInterval(() => {
      indexRef.current++;
      setDisplayed(text.slice(0, indexRef.current));
      if (indexRef.current >= text.length) clearInterval(timer);
    }, speed);

    return () => clearInterval(timer);
  }, [text, baseSpeed]);

  return displayed;
}
