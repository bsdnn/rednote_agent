import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core.logging_config import setup_logging, request_id_var
from .api.routes import router
from .core.config import settings

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .services.rag.service import query_product_database
    from .services.memory_service import init_db
    from .services.deepseek_client import get_client, close_client

    await init_db()
    logger.info("Memory database initialized")

    logger.info("Warming up RAG service (version=%s)...", settings.RAG_VERSION)
    await query_product_database("预热")
    logger.info("RAG warmup complete")

    get_client()
    logger.info("DeepSeek client initialized")

    yield

    await close_client()
    logger.info("Application shutdown complete")


app = FastAPI(title="Xiaohongshu AI Generator API", version="2.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:8])
    token = request_id_var.set(req_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    request_id_var.reset(token)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)


_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_static_dir / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        index = _static_dir / "index.html"
        return FileResponse(str(index))
