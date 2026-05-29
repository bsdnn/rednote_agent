"""Run RAG v2 ablation eval.

Mirrors scripts/ab_planning_experiment.py: paired design, incremental JSONL output,
resume on restart, --dry-run for first 5 queries.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.persona import UserPersona
from backend.services.deepseek_client import get_client, close_client
from scripts.rag_eval.configs import CONFIG_SPECS, build_query_fn
from scripts.rag_eval.metrics import recall_at_k, mrr, has_forbidden, judge_faithfulness

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GOLD = ROOT / "scripts" / "rag_eval" / "gold_queries.jsonl"
RESULTS_DIR = ROOT / "scripts" / "rag_eval" / "results"


def _load_gold():
    return [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_doc_ids(context: str) -> list[str]:
    """Parse '[匹配N] ...' or chunk text — extract any prod_/ing_/post_ IDs mentioned.

    Pipeline's format_hits now embeds [doc_id] thanks to Task 16 Step 2 edits.
    """
    return re.findall(r"(prod_\d+|ing_\w+|post_\d+)", context)


async def _run_config(spec, gold_records: list[dict], judge_client, out_path: Path, done_qids: set[str]):
    from scripts.rag_eval.metrics import judge_relevancy_li
    query_fn = build_query_fn(spec)

    for rec in gold_records:
        if rec["qid"] in done_qids:
            continue

        persona = UserPersona(**rec["persona"]) if rec.get("persona") else None

        t0 = time.monotonic()
        try:
            context = await query_fn(rec["query"], persona)
            error = None
        except Exception as e:
            context, error = "", str(e)
        elapsed = time.monotonic() - t0

        retrieved_ids = _extract_doc_ids(context)
        gold_ids = set(rec.get("relevant_doc_ids", []))
        forbidden = set(rec.get("must_not_contain", []))

        faith = await judge_faithfulness(judge_client, rec["query"], context) if context else None
        li_relevancy = await judge_relevancy_li(rec["query"], context) if context else None

        out = {
            "config": spec.name,
            "qid": rec["qid"],
            "query": rec["query"],
            "category": rec["category"],
            "retrieved_ids": retrieved_ids,
            "recall_at_3": recall_at_k(retrieved_ids, gold_ids, 3),
            "recall_at_10": recall_at_k(retrieved_ids, gold_ids, 10),
            "mrr": mrr(retrieved_ids, gold_ids),
            "has_forbidden": has_forbidden(retrieved_ids, forbidden),
            "faithfulness": faith.get("faithfulness") if faith else None,
            "faithfulness_reason": faith.get("reason") if faith else None,
            "li_relevancy_pass": li_relevancy.get("pass") if li_relevancy else None,
            "li_relevancy_feedback": li_relevancy.get("feedback") if li_relevancy else None,
            "elapsed_seconds": elapsed,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        logger.info(
            "[%s] %s  R@3=%.2f  MRR=%.2f  faith=%s  li=%s  %.2fs",
            spec.name, rec["qid"], out["recall_at_3"], out["mrr"],
            out["faithfulness"], out["li_relevancy_pass"], elapsed,
        )


async def main_async(args):
    gold = _load_gold()
    if args.dry_run:
        gold = gold[:5]
        logger.info("DRY RUN: first %d queries only", len(gold))

    configs = [c for c in CONFIG_SPECS if args.config == "all" or c.name == args.config]
    if not configs:
        sys.exit(f"unknown config: {args.config}")

    judge_client = get_client()
    date_str = datetime.now().strftime("%Y-%m-%d")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in configs:
        out_path = RESULTS_DIR / f"{date_str}_{spec.name}.jsonl"
        done_qids = set()
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done_qids.add(json.loads(line)["qid"])
            logger.info("[%s] resume: %d done", spec.name, len(done_qids))

        await _run_config(spec, gold, judge_client, out_path, done_qids)

    await close_client()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="all", help="C0|C1|C2|C3|C4|all")
    p.add_argument("--n", type=int, default=None, help="reserved")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm", action="store_true", help="required for full run")
    args = p.parse_args()
    if not args.dry_run and not args.confirm:
        sys.exit("Full runs require --confirm to avoid accidental cost. Use --dry-run first.")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
