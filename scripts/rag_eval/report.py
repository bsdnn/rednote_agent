"""Aggregate scripts/rag_eval/results/*.jsonl into a Markdown report.

Usage: python scripts/rag_eval/report.py [--date YYYY-MM-DD]
Output: docs/testing/<date>-rag-v3-report.md
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

    sections = [f"# RAG v3 Eval Report — {date}\n"]
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

    # Dual-judge consistency: how often does LI RelevancyEvaluator (pass/fail)
    # agree with custom faithfulness (>= 7 = "pass")
    sections.append("\n## Dual-Judge Consistency (Custom-1to10 vs LI-RelevancyEvaluator)\n")
    sections.append("Custom score >= 7 treated as PASS. Cohen's κ measures agreement beyond chance.\n")
    sections.append("| Config | Both PASS | Both FAIL | Custom PASS / LI FAIL | Custom FAIL / LI PASS | Cohen κ |")
    sections.append("|---|---|---|---|---|---|")
    for config in all_aggs:
        recs, _ = all_aggs[config]
        both_pass = both_fail = custom_only = li_only = 0
        for r in recs:
            f = r.get("faithfulness")
            lp = r.get("li_relevancy_pass")
            if f is None or lp is None:
                continue
            custom_pass = f >= 7
            if custom_pass and lp:
                both_pass += 1
            elif (not custom_pass) and (not lp):
                both_fail += 1
            elif custom_pass and (not lp):
                custom_only += 1
            else:
                li_only += 1
        total = both_pass + both_fail + custom_only + li_only
        if total == 0:
            sections.append(f"| {config} | 0 | 0 | 0 | 0 | N/A |")
            continue
        agreement = (both_pass + both_fail) / total
        p_custom_pass = (both_pass + custom_only) / total
        p_li_pass = (both_pass + li_only) / total
        p_e = p_custom_pass * p_li_pass + (1 - p_custom_pass) * (1 - p_li_pass)
        kappa = (agreement - p_e) / (1 - p_e) if p_e < 1.0 else 1.0
        sections.append(
            f"| {config} | {both_pass} | {both_fail} | {custom_only} | {li_only} | {kappa:+.3f} |"
        )

    # Failure analysis: low-faithfulness reasons grouped by category
    sections.append("\n## Failure Analysis (faithfulness ≤ 5)\n")
    any_failure = False
    for config in all_aggs:
        recs, _ = all_aggs[config]
        low = [r for r in recs
               if r.get("faithfulness") is not None and r["faithfulness"] <= 5
               and r.get("faithfulness_reason")]
        if not low:
            continue
        any_failure = True
        sections.append(f"### {config} — {len(low)} low-score records\n")
        by_cat = defaultdict(list)
        for r in low:
            by_cat[r.get("category", "uncategorized")].append(r)
        for cat in sorted(by_cat.keys()):
            sections.append(f"**{cat}** ({len(by_cat[cat])} records)")
            for r in by_cat[cat][:2]:
                sections.append(f"- `{r['qid']}` _{r['query']}_ (faith={r['faithfulness']})")
                sections.append(f"  > {r['faithfulness_reason']}")
            sections.append("")
    if not any_failure:
        sections.append("_No low-faithfulness records (≤5) with captured reason._\n")

    return "\n".join(sections) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = p.parse_args()

    md = build_report(args.date)
    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    out = DOCS_OUT / f"{args.date}-rag-v3-report.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
