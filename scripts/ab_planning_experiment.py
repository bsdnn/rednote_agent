"""A/B experiment: does _planning_phase improve agent output enough
to justify its extra cost?

Paired design: each query is run once with planning ON (arm A) and
once with planning OFF (arm B), in randomized order. Captures:
- reflection scores (post-hoc, applied to final draft)
- token usage (prompt + completion + total)
- iteration count and wall-clock time

Incremental JSONL output makes the run resumable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.agent import generate_rednote, _reflection_phase
from backend.models.request import GenerateRequest, Tone
from backend.models.response import GenerateResponse
from backend.services.deepseek_client import get_client

QUERY_FILE = ROOT / "scripts" / "ab_queries.json"
DEFAULT_OUT_JSONL = ROOT / "scripts" / "ab_results.jsonl"
DEFAULT_REPORT_MD = ROOT / "docs" / "testing" / "2026-05-18-ab-report.md"


def _empty_record(query_id: int, query: str, arm: str) -> dict:
    return {
        "query_id": query_id,
        "query": query,
        "arm": arm,
        "success": False,
        "error": None,
        "reflection_min_score": None,
        "reflection_virality": None,
        "reflection_tone": None,
        "reflection_accuracy": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "iterations": 0,
        "wall_clock_seconds": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _run_one(query_id: int, query: str, arm: str) -> dict:
    record = _empty_record(query_id, query, arm)
    enable_planning = (arm == "A")
    final_draft: dict | None = None

    t0 = time.monotonic()
    try:
        req = GenerateRequest(query=query, tone=Tone.lively, max_iterations=5)
        async for ev in generate_rednote(req, enable_planning=enable_planning):
            if ev["event"] == "agent_thinking":
                it = ev["data"].get("iteration")
                if isinstance(it, int):
                    record["iterations"] = max(record["iterations"], it)
            elif ev["event"] == "complete":
                final_draft = ev["data"]
                record["success"] = True
            elif ev["event"] == "token_usage":
                record["prompt_tokens"] = ev["data"]["prompt_tokens"]
                record["completion_tokens"] = ev["data"]["completion_tokens"]
                record["total_tokens"] = ev["data"]["total_tokens"]
            elif ev["event"] == "error":
                record["error"] = ev["data"].get("message")
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"

    record["wall_clock_seconds"] = round(time.monotonic() - t0, 3)

    if final_draft is not None:
        try:
            draft_obj = GenerateResponse.model_validate(final_draft)
            critique = await _reflection_phase(get_client(), draft_obj)
            record["reflection_min_score"] = critique.get("min_score")
            record["reflection_virality"] = critique.get("virality_score")
            record["reflection_tone"] = critique.get("tone_match_score")
            record["reflection_accuracy"] = critique.get("accuracy_score")
        except Exception as e:
            record["error"] = (record["error"] or "") + f" | post-hoc reflection: {e}"

    return record


def _load_done_pairs(jsonl_path: Path) -> set[tuple[int, str]]:
    if not jsonl_path.exists():
        return set()
    done: set[tuple[int, str]] = set()
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("success"):
                    done.add((rec["query_id"], rec["arm"]))
            except json.JSONDecodeError:
                continue
    return done


def _build_run_plan(queries: list[dict], n: int, done: set[tuple[int, str]],
                    seed: int) -> list[tuple[int, str, str]]:
    """Return [(query_id, query, arm)] in randomized within-pair order."""
    rng = random.Random(seed)
    plan: list[tuple[int, str, str]] = []
    for q in queries[:n]:
        arms = ["A", "B"]
        rng.shuffle(arms)
        for arm in arms:
            if (q["id"], arm) not in done:
                plan.append((q["id"], q["query"], arm))
    return plan


async def _run_experiment(n: int, jsonl_out: Path, seed: int) -> list[dict]:
    queries = json.loads(QUERY_FILE.read_text(encoding="utf-8"))["queries"]
    done = _load_done_pairs(jsonl_out)
    plan = _build_run_plan(queries, n, done, seed)

    print(f"[ab] {len(plan)} runs to execute "
          f"({len(done)} already done, {n * 2 - len(done) - len(plan)} skipped)")
    print(f"[ab] streaming results to {jsonl_out}")

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for i, (qid, qtext, arm) in enumerate(plan, 1):
        print(f"[ab] {i}/{len(plan)}  qid={qid:>2}  arm={arm}  '{qtext[:30]}...'", flush=True)
        rec = await _run_one(qid, qtext, arm)
        with jsonl_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        all_records.append(rec)
        status = "ok " if rec["success"] else "FAIL"
        score = rec["reflection_min_score"]
        tokens = rec["total_tokens"]
        print(f"        -> {status} score={score} tokens={tokens} "
              f"iter={rec['iterations']} t={rec['wall_clock_seconds']}s")

    return all_records


def _summarize(jsonl_path: Path) -> dict:
    """Read all records and compute paired comparison summary."""
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    by_pair: dict[int, dict[str, dict]] = {}
    for r in records:
        by_pair.setdefault(r["query_id"], {})[r["arm"]] = r

    paired = [(qid, p["A"], p["B"]) for qid, p in by_pair.items()
              if "A" in p and "B" in p and p["A"]["success"] and p["B"]["success"]]

    def arm_stat(field: str, arm: str) -> tuple[float | None, float | None]:
        vals = [r.get(field) for r in records if r["arm"] == arm]
        vals = [v for v in vals if v is not None]
        if not vals:
            return (None, None)
        return (round(mean(vals), 3), round(stdev(vals), 3) if len(vals) > 1 else 0.0)

    summary = {
        "total_records": len(records),
        "paired_count": len(paired),
        "successes_A": sum(1 for r in records if r["arm"] == "A" and r["success"]),
        "successes_B": sum(1 for r in records if r["arm"] == "B" and r["success"]),
        "failures": [r for r in records if not r["success"]],
    }

    for field in ("reflection_min_score", "total_tokens", "iterations",
                  "wall_clock_seconds"):
        for arm in ("A", "B"):
            m, sd = arm_stat(field, arm)
            summary[f"{field}_{arm}_mean"] = m
            summary[f"{field}_{arm}_stdev"] = sd

    def paired_diff(field: str) -> dict:
        diffs = [(qid, a[field], b[field], a[field] - b[field])
                 for qid, a, b in paired
                 if a[field] is not None and b[field] is not None]
        if not diffs:
            return {"n": 0, "mean_diff": None, "a_wins": 0, "ties": 0, "b_wins": 0}
        return {
            "n": len(diffs),
            "mean_diff": round(mean(d[3] for d in diffs), 3),
            "a_wins": sum(1 for d in diffs if d[3] > 0),
            "ties":   sum(1 for d in diffs if d[3] == 0),
            "b_wins": sum(1 for d in diffs if d[3] < 0),
            "per_query": diffs,
        }

    summary["paired_score"]      = paired_diff("reflection_min_score")
    summary["paired_tokens"]     = paired_diff("total_tokens")
    summary["paired_iterations"] = paired_diff("iterations")
    summary["paired_wall_clock"] = paired_diff("wall_clock_seconds")
    return summary


def _write_report(summary: dict, report_path: Path, n: int) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v):
        return "—" if v is None else v

    lines: list[str] = []
    lines.append(f"# A/B 实验报告：Planning 阶段是否值得？")
    lines.append("")
    lines.append(f"**生成时间：** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**设计：** 配对设计，N={n} 条 query，每条在 A（planning=ON）"
                 f"和 B（planning=OFF）各跑一次。")
    lines.append(f"**实际数据：** 总记录 {summary['total_records']}，"
                 f"成功配对 {summary['paired_count']}，"
                 f"A 成功 {summary['successes_A']}，B 成功 {summary['successes_B']}。")
    lines.append("")
    lines.append("## 汇总指标")
    lines.append("")
    lines.append("| 指标 | A (planning) mean ± sd | B (no planning) mean ± sd |")
    lines.append("|---|---|---|")
    for field, label in [
        ("reflection_min_score", "Reflection 最低分"),
        ("total_tokens", "总 tokens"),
        ("iterations", "迭代数"),
        ("wall_clock_seconds", "墙钟秒数"),
    ]:
        lines.append(
            f"| {label} | {fmt(summary[f'{field}_A_mean'])} ± "
            f"{fmt(summary[f'{field}_A_stdev'])} | "
            f"{fmt(summary[f'{field}_B_mean'])} ± "
            f"{fmt(summary[f'{field}_B_stdev'])} |"
        )
    lines.append("")
    lines.append("## 配对差值（A − B，正数 = A 更高）")
    lines.append("")
    lines.append("| 指标 | n | 配对差均值 | A 胜 | 平 | B 胜 |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in [
        ("paired_score", "Reflection 最低分"),
        ("paired_tokens", "总 tokens"),
        ("paired_iterations", "迭代数"),
        ("paired_wall_clock", "墙钟秒数"),
    ]:
        d = summary[key]
        lines.append(
            f"| {label} | {d['n']} | {fmt(d['mean_diff'])} | "
            f"{d['a_wins']} | {d['ties']} | {d['b_wins']} |"
        )
    lines.append("")

    failures = summary["failures"]
    if failures:
        lines.append("## 失败案例")
        lines.append("")
        for f in failures:
            lines.append(f"- qid={f['query_id']} arm={f['arm']} — {f.get('error')}")
        lines.append("")

    lines.append("## 诚实结论段")
    lines.append("")
    lines.append("**已知局限：**")
    lines.append("")
    lines.append("- **Self-eval bias：** 评分用的是同一个 DeepSeek 模型做 reflection，"
                 "存在自评偏高的可能。本报告不宣称\"质量\"差异，只宣称\"系统自评分\"差异。")
    lines.append(f"- **样本量 {n} 条 query** 不足以做严格的显著性检验。本报告只报"
                 "趋势、配对差均值、胜率，不报 p-value。")
    lines.append("- **单评估源：** 没有引入独立 judge 模型，因此无法消除评估偏差。")
    lines.append("- **Query 集人工编写：** 不能外推到所有用户场景。")
    lines.append("")
    lines.append("**怎么看这份数据：**")
    lines.append("")
    lines.append("- 如果 A 路 reflection 分均值 > B 路且配对胜率 > 60% → planning 有正向效果，"
                 "值得保留；")
    lines.append("- 如果 A 路总 tokens 显著高于 B 路且 reflection 分无优势 → planning 是纯"
                 "成本，建议移除；")
    lines.append("- 如果两路指标接近 → planning 价值未证，下一步应该用更大样本 + "
                 "外部 judge 重测。")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ab] report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Planning A/B experiment.")
    parser.add_argument("--n", type=int, default=30,
                        help="Number of queries (default 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use only 3 queries; skip cost check")
    parser.add_argument("--confirm", action="store_true",
                        help="Required if --n > 10")
    parser.add_argument("--resume", action="store_true",
                        help="Skip query/arm pairs already present in --out")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_JSONL,
                        help="JSONL output file")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_MD,
                        help="Markdown report file")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for arm-order shuffling")
    args = parser.parse_args()

    if args.dry_run:
        n = 3
    else:
        n = args.n
        if n > 10 and not args.confirm:
            print(f"refusing to run {n} queries × 2 arms without --confirm "
                  f"(this will cost real DeepSeek tokens, est. ${n * 0.06:.2f}-${n * 0.10:.2f})")
            sys.exit(1)

    if not args.resume and args.out.exists():
        args.out.unlink()

    print(f"[ab] starting: n={n}, planning A on, B off, paired")
    print(f"[ab] estimated cost: ${n * 0.06:.2f}-${n * 0.10:.2f} USD")
    asyncio.run(_run_experiment(n, args.out, args.seed))
    summary = _summarize(args.out)
    _write_report(summary, args.report, n)


if __name__ == "__main__":
    main()
