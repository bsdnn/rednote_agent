from collections import defaultdict
from time import time

from fastapi import HTTPException, Request

from ..core.config import settings

_request_times: dict[str, list[float]] = defaultdict(list)


async def rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = time()
    window = 60

    _request_times[client_ip] = [
        t for t in _request_times[client_ip] if now - t < window
    ]

    if len(_request_times[client_ip]) >= settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
            headers={"Retry-After": str(window)},
        )

    _request_times[client_ip].append(now)
