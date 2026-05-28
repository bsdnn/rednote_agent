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

# 50 mainstream 小红书 skincare ingredient slugs
INGREDIENT_SLUGS = [
    # 经典酸 / 维C / A醇家族
    "niacinamide", "hyaluronic_acid", "sodium_hyaluronate", "retinol", "retinal",
    "vitamin_c", "vc_derivative_etvc", "vc_derivative_thd",
    "salicylic_acid", "glycolic_acid", "mandelic_acid", "lactic_acid",
    # 美白家族
    "tranexamic_acid", "arbutin", "alpha_arbutin", "symwhite_377",
    "glutathione", "ergothioneine", "ellagic_acid",
    # 抗老 / 肽 / 胶原
    "pro_xylane", "copper_peptides", "argireline_hexapeptide",
    "matrixyl_palmitoyl_pentapeptide", "recombinant_collagen", "collagen_tripeptide",
    "peptide_egf",
    # 屏障 / 保湿
    "ceramides", "ceramide_np", "squalane", "polyglutamic_acid", "beta_glucan",
    # 舒缓 / 抗炎
    "centella_madecassoside", "centella_asiatica", "bisabolol", "allantoin",
    "panthenol", "azelaic_acid", "ectoin", "licorice_extract",
    # 抗氧化 / 经典 vitamin
    "vitamin_e", "ferulic_acid", "alpha_lipoic_acid", "coq10",
    # 发酵 / 提取物 — 真热门
    "galactomyces", "tea_polyphenols", "bird_nest_extract", "sialic_acid",
    "propolis_extract", "salicin",
    # 闭塞剂
    "petrolatum",
]
assert len(INGREDIENT_SLUGS) == 50, f"expected 50, got {len(INGREDIENT_SLUGS)}"


PROMPT_TEMPLATE = """你是小红书护肤博主+成分党。生成一条护肤成分百科,严格输出 JSON。

成分英文 slug: {slug}

写作要求(必须满足):
1. content (180-250字):必须包含
   - 成分中文名 + 常见别名或 INCI 名
   - 核心机制(一句话原理,如"抑制酪氨酸酶活性")
   - 常用浓度范围(如"2-5%")或推荐有效剂量
   - 使用时机(早C晚A / 仅夜间 / AM&PM 均可)
   - 1-2 个在售产品例子(如"修丽可 CE 精华、HABA G露、SK-II 神仙水")
   口语化但有干货,小红书风格但不浮夸

2. effects (3-5个):从下列固定词表中选,可额外自定义最多 1 个
   ["美白","淡斑","抗老","抗皱","保湿","修护","屏障","抗炎","舒缓","控油","抗痘","紧致","提亮","抗氧化","祛黄"]

3. concerns (2-4条):每条必须具体可执行
   - 至少一条层叠冲突(如"不可与高浓度VC同时使用")
   - 至少一条肤质/孕期/光敏具体警告
   - 浓度起点建议(如"建议从0.025% A醇起步")
   禁止"敏感肌慎用"这种空话

4. age_groups: 从 ["18-24","25-30","31-40","41+"] 选,**最多 3 个**(避免全选)

严格输出 JSON,不要 markdown 围栏,不要解释:
{{"slug":"{slug}","name":"中文名","content":"...","effects":["..."],"concerns":["..."],"age_groups":["..."]}}
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
