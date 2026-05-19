import aiosqlite
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


async def init_db() -> None:
    """Create tables if they don't exist. Call once at application startup."""
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT PRIMARY KEY,
                persona  TEXT,
                last_seen TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS copy_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                query      TEXT,
                title      TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.commit()


async def recall_user_history(user_id: str) -> str:
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT persona, last_seen FROM user_memory WHERE user_id = ?", (user_id,)
        )
        user_row = await cur.fetchone()

        cur = await db.execute(
            "SELECT query, title, created_at FROM copy_history "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT 3",
            (user_id,),
        )
        history_rows = await cur.fetchall()

    if not user_row and not history_rows:
        return f"用户「{user_id}」暂无历史记录，这是首次使用，请根据当前请求生成内容。"

    parts: list[str] = []
    if user_row and user_row["persona"]:
        parts.append(f"用户画像：{user_row['persona']}（最近访问：{user_row['last_seen']}）")
    if history_rows:
        entries = "；".join(
            f"「{r['query']}」→《{r['title']}》" for r in history_rows
        )
        parts.append(f"最近生成记录：{entries}")

    return "\n".join(parts)


async def save_copy_result(
    user_id: str, query: str, title: str, persona_json: str | None = None
) -> None:
    async with aiosqlite.connect(_DB_PATH) as db:
        if persona_json:
            await db.execute(
                """
                INSERT INTO user_memory (user_id, persona, last_seen)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(user_id) DO UPDATE
                SET persona = excluded.persona, last_seen = excluded.last_seen
                """,
                (user_id, persona_json),
            )
        else:
            await db.execute(
                """
                INSERT INTO user_memory (user_id, last_seen)
                VALUES (?, datetime('now'))
                ON CONFLICT(user_id) DO UPDATE SET last_seen = excluded.last_seen
                """,
                (user_id,),
            )
        await db.execute(
            "INSERT INTO copy_history (user_id, query, title) VALUES (?, ?, ?)",
            (user_id, query, title),
        )
        await db.commit()
