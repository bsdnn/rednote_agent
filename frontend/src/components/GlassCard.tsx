import type { ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  glow?: boolean;
}

export default function GlassCard({ children, className = "", glow = false }: GlassCardProps) {
  return (
    <div className={`glass-card ${glow ? "glow-red" : ""} ${className}`}>
      {children}
    </div>
  );
}
