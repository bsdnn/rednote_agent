import logging
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from ..core.config import settings

logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    async def chat(self, messages: list[dict], tools: list | None = None) -> dict:
        payload: dict = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.debug("Sending chat request with %d messages", len(messages))
        response = await self._client.post(settings.DEEPSEEK_API_URL, json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self._client.aclose()


# Module-level singleton — shared across all requests
_client: DeepSeekClient | None = None


def get_client() -> DeepSeekClient:
    global _client
    if _client is None:
        _client = DeepSeekClient()
        logger.info("DeepSeek client singleton created")
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
        logger.info("DeepSeek client closed")
