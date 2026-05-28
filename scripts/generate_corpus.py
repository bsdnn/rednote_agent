"""LLM-synthesized corpus generator.

Subcommands:
    ingredients  - generate ~50 ingredient wiki entries
    posts        - generate ~250 Xiaohongshu-style posts (Task 13 — stub)

Output written to backend/data/corpus/{ingredients,posts}.json.
All entries are tagged metadata.source="synthesized" by the loader.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services.deepseek_client import get_client, close_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "backend" / "data" / "corpus"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 50 canonical skincare ingredient slugs
INGREDIENT_SLUGS = [
    "niacinamide", "vitamin_c", "retinol", "hyaluronic_acid", "salicylic_acid",
    "glycolic_acid", "lactic_acid", "azelaic_acid", "kojic_acid", "arbutin",
    "tranexamic_acid", "alpha_arbutin", "ceramides", "peptides", "panthenol",
    "centella_asiatica", "allantoin", "bisabolol", "squalane", "snail_mucin",
    "bakuchiol", "alpha_hydroxy_acids", "beta_hydroxy_acids", "polyhydroxy_acids",
    "glutathione", "resveratrol", "ferulic_acid", "vitamin_e", "coq10",
    "ectoin", "madecassoside", "tiger_grass", "honeysuckle", "green_tea",
    "licorice_extract", "mulberry_extract", "pearl_powder", "collagen",
    "elastin", "amino_acids", "zinc", "copper_peptides", "edelweiss",
    "rosehip_oil", "jojoba_oil", "argan_oil", "marula_oil", "rosemary",
    "tea_tree", "lavender_oil",
]


PROMPT_TEMPLATE = """生成一条护肤成分百科条目,严格输出 JSON。

成分英文 slug: {slug}

要求:
- 中文名 + 简介(100-150字),口语化但准确
- effects: 该成分的核心功效列表(中文,3-5个,如 "美白" "保湿" "抗老" "舒缓" "控油" 等)
- concerns: 使用注意事项列表(中文,1-3条,如 "敏感肌慎用" "孕期禁用" 等;无禁忌则返回空数组)
- age_groups: 适合年龄段(从 ["18-24", "25-30", "31-40", "41+"] 中选,可多选)

输出格式(严格 JSON,不要任何解释文字,不要 markdown 围栏):
{{"slug": "{slug}", "name": "中文名", "content": "百科介绍...", "effects": ["..."], "concerns": ["..."], "age_groups": ["..."]}}
"""


async def _gen_one_ingredient(client, slug: str) -> dict | None:
    """Call DeepSeek to generate one ingredient entry.

    NOTE: DeepSeekClient.chat() hardcodes model='deepseek-chat', temperature=0.7,
    response_format={'type': 'json_object'} — exactly what we want here.
    """
    try:
        result = await client.chat(messages=[
            {"role": "user", "content": PROMPT_TEMPLATE.format(slug=slug)}
        ])
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        logger.error("Failed slug=%s: %s", slug, e)
        return None


async def cmd_ingredients(args):
    random.seed(42)
    client = get_client()
    out_path = DATA_DIR / "ingredients.json"
    existing = []
    if out_path.exists() and not args.force:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        done_slugs = {r["slug"] for r in existing}
        logger.info("Resume: %d already done", len(done_slugs))
        todo = [s for s in INGREDIENT_SLUGS if s not in done_slugs]
    else:
        todo = list(INGREDIENT_SLUGS)

    results = list(existing)
    for slug in todo:
        record = await _gen_one_ingredient(client, slug)
        if record:
            results.append(record)
            # incremental write — preserves progress if killed mid-run
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("OK %s (%d/%d)", slug, len(results), len(INGREDIENT_SLUGS))

    await close_client()
    logger.info("Done: %d ingredients written to %s", len(results), out_path)


async def cmd_posts(args):
    # Implemented in Task 13.
    raise NotImplementedError("cmd_posts is implemented in Task 13")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingredients")
    p_ing.add_argument("--force", action="store_true", help="overwrite existing")
    p_post = sub.add_parser("posts")
    p_post.add_argument("--n", type=int, default=250)
    p_post.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.cmd == "ingredients":
        asyncio.run(cmd_ingredients(args))
    elif args.cmd == "posts":
        asyncio.run(cmd_posts(args))


if __name__ == "__main__":
    main()
