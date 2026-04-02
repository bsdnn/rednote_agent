from __future__ import annotations
import json
import logging
from time import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..api.dependencies import rate_limit
from ..core.agent import generate_rednote, refine_rednote
from ..models.request import GenerateRequest, RefineRequest
from ..services.rag_service import PRODUCTS

logger = logging.getLogger(__name__)
router = APIRouter()

_start_time = time()


@router.get("/api/health")
async def health():
    return {
        "status": "ok",
        "rag_products": len(PRODUCTS),
        "uptime_seconds": round(time() - _start_time),
        "model": "deepseek-chat",
    }


@router.post("/api/generate", dependencies=[Depends(rate_limit)])
async def api_generate(request: GenerateRequest, http_request: Request):
    logger.info("POST /api/generate query='%s'", request.query)

    async def event_stream():
        async for event in generate_rednote(request):
            if await http_request.is_disconnected():
                break
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"], ensure_ascii=False),
            }

    return EventSourceResponse(event_stream())


@router.post("/api/refine", dependencies=[Depends(rate_limit)])
async def api_refine(request: RefineRequest, http_request: Request):
    logger.info("POST /api/refine instruction='%s'", request.refinement_instruction)

    async def event_stream():
        async for event in refine_rednote(request):
            if await http_request.is_disconnected():
                break
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"], ensure_ascii=False),
            }

    return EventSourceResponse(event_stream())
