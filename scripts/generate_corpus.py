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


POST_TOPICS = [
    # 控油/痘 (3)
    "油皮控油精华推荐", "祛痘印成分对比", "黑头粉刺真实使用",
    # 干/敏 (3)
    "干皮急救面膜", "敏感肌舒缓修护", "屏障受损用什么",
    # 美白/抗老 (4)
    "美白淡斑成分横评", "377熊果苷谁更稳", "抗老紧致28天打卡", "祛黄熬夜急救",
    # 场景/痛点 (5)
    "毛孔粗大缩毛孔", "熬夜暗沉急救", "换季泛红舒缓", "晒后修复方案", "孕期可用安全清单",
    # 人群细分 (5)
    "学生党平价100元内", "通勤快速3步护肤", "30岁后抗老进阶", "敏感肌痘印双困扰", "男生入门护肤",
]


POST_PROMPT = """你是小红书护肤分享博主,粉丝量50w+。生成一条种草/避雷类护肤文案,严格输出 JSON。

post_id: {pid}
主题: {topic}

写作要求(必须满足):
1. title (12-18字):带数字/对比/反问,典型小红书标题风格,可带1-2个emoji
2. content (180-280字):必须包含
   - 痛点开场(具体场景,如"上班族下午脸出油到爆")
   - 至少1-2个具体成分(从[烟酰胺,玻尿酸,A醇,维C,377,玻色因,二裂酵母,蓝铜肽,神经酰胺,水杨酸]中选)
   - 实际产品例(如"修丽可CE精华、SK-II神仙水")
   - 使用timeline(如"用了21天")或具体细节(浓度/搭配/早晚)
   - 收尾给具体行动建议
   口语化、有emoji、惊叹号,但不要全篇情绪化
3. tone(固定5选):["活泼甜美","专业种草","温柔治愈","幽默搞笑","精英范儿"]
4. linked_products (1-3个,可空): 真实在售产品中文名,如["修丽可CE精华","SK-II神仙水"]

严格输出 JSON,不要markdown围栏:
{{"post_id":"{pid}","title":"...","content":"...","tone":"...","linked_products":["..."]}}
"""


async def _gen_one_post(client, pid: int, topic: str) -> dict | None:
    try:
        result = await client.chat(messages=[
            {"role": "user", "content": POST_PROMPT.format(pid=pid, topic=topic)}
        ])
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        logger.error("post %d failed: %s", pid, e)
        return None


async def cmd_posts(args):
    random.seed(42)
    client = get_client()
    out_path = DATA_DIR / "posts.json"
    existing = []
    done_ids: set[str] = set()
    if out_path.exists() and not args.force:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        done_ids = {str(r["post_id"]) for r in existing}
        logger.info("Resume: %d posts already done", len(done_ids))
    elif out_path.exists() and args.force:
        out_path.unlink()
        logger.info("Force flag: deleted existing posts.json")

    n = args.n
    results = list(existing)
    sem = asyncio.Semaphore(5)
    lock = asyncio.Lock()

    async def _one(pid: int):
        if str(pid) in done_ids:
            return
        topic = random.choice(POST_TOPICS)
        async with sem:
            rec = await _gen_one_post(client, pid, topic)
            if rec:
                async with lock:
                    results.append(rec)
                    out_path.write_text(
                        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                logger.info("post_%d (%s) %d/%d", pid, topic, len(results), n)

    await asyncio.gather(*[_one(i) for i in range(n)])
    await close_client()
    logger.info("Done: %d posts written to %s", len(results), out_path)


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
