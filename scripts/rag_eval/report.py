"""Aggregate scripts/rag_eval/results/*.jsonl into a Markdown report.

Usage: python scripts/rag_eval/report.py [--date YYYY-MM-DD]
Output: docs/testing/<date>-rag-v2-report.md
"""
from __future__ import annotations
import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "scripts" / "rag_eval" / "results"
DOCS_OUT = ROOT / "docs" / "testing"


def _load_config(jsonl_path: Path) -> list[dict]:
    return [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _agg(records: list[dict]) -> dict:
    if not records: return {}
    def _mean(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return statistics.mean(vals) if vals else None
    return {
        "n": len(records),
        "recall_at_3": _mean("recall_at_3"),
        "recall_at_10": _mean("recall_at_10"),
        "mrr": _mean("mrr"),
        "faithfulness": _mean("faithfulness"),
        "elapsed_seconds": _mean("elapsed_seconds"),
        "forbidden_violations": sum(1 for r in records if r.get("has_forbidden")),
    }


def _agg_by_category(records: list[dict]) -> dict[str, dict]:
    buckets = defaultdict(list)
    for r in records:
        buckets[r.get("category", "uncategorized")].append(r)
    return {cat: _agg(recs) for cat, recs in buckets.items()}


def build_report(date: str) -> str:
    files = sorted(RESULTS.glob(f"{date}_C*.jsonl"))
    if not files:
        return f"No results found for date {date} in {RESULTS}/"

    sections = [f"# RAG v2 Eval Report — {date}\n"]
    sections.append("## Overall Metrics\n")
    sections.append("| Config | N | Recall@3 | Recall@10 | MRR | Faithfulness | Avg sec | Forbidden |")
    sections.append("|---|---|---|---|---|---|---|---|")

    all_aggs = {}
    for f in files:
        config = f.stem.split("_")[-1]
        recs = _load_config(f)
        agg = _agg(recs)
        all_aggs[config] = (recs, agg)
        sections.append(
            f"| {config} | {agg['n']} | {agg['recall_at_3']:.3f} | {agg['recall_at_10']:.3f} | "
            f"{agg['mrr']:.3f} | {agg['faithfulness'] if agg['faithfulness'] else 'N/A'} | "
            f"{agg['elapsed_seconds']:.2f} | {agg['forbidden_violations']} |"
        )

    sections.append("\n## Per-Category Breakdown (Recall@3)\n")
    sections.append("| Category | " + " | ".join(all_aggs.keys()) + " |")
    sections.append("|---|" + "---|" * len(all_aggs))
    cats = set()
    for recs, _ in all_aggs.values():
        cats.update(r.get("category", "uncategorized") for r in recs)
    for cat in sorted(cats):
        row = [f"| {cat}"]
        for config in all_aggs:
            recs, _ = all_aggs[config]
            cat_recs = [r for r in recs if r.get("category") == cat]
            mean_r3 = statistics.mean([r["recall_at_3"] for r in cat_recs]) if cat_recs else 0.0
            row.append(f" {mean_r3:.3f}")
        sections.append("|".join(row) + " |")

    sections.append("\n## Delta vs Baseline (C0)\n")
    if "C0" in all_aggs:
        base = all_aggs["C0"][1]
        sections.append("| Config | ΔRecall@3 | ΔMRR | ΔFaithfulness |")
        sections.append("|---|---|---|---|")
        for config in all_aggs:
            if config == "C0": continue
            cur = all_aggs[config][1]
            d_r3 = cur["recall_at_3"] - base["recall_at_3"]
            d_mrr = cur["mrr"] - base["mrr"]
            d_f = ((cur["faithfulness"] or 0) - (base["faithfulness"] or 0)) if base["faithfulness"] else None
            sections.append(f"| {config} | {d_r3:+.3f} | {d_mrr:+.3f} | {f'{d_f:+.2f}' if d_f is not None else 'N/A'} |")

    return "\n".join(sections) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = p.parse_args()

    md = build_report(args.date)
    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    out = DOCS_OUT / f"{args.date}-rag-v2-report.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
