import asyncio
import time
from typing import Callable
import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


class SemanticCache:
    def __init__(
        self,
        embedding_fn: Callable[[str], np.ndarray],
        threshold: float = 0.92,
        max_size: int = 256,
        ttl_seconds: int = 3600,
    ):
        self._embed = embedding_fn
        self._threshold = threshold
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._keys: list[str] = []
        self._embeds: list[np.ndarray] = []
        self._values: list[str] = []
        self._timestamps: list[float] = []
        self._access: list[float] = []  # for LRU
        self._lock = asyncio.Lock()
        self._stats = {
            "exact_hits": 0, "semantic_hits": 0, "misses": 0,
            "evictions_lru": 0, "evictions_ttl": 0,
        }

    def _evict_expired(self) -> None:
        now = time.time()
        keep_idx = [i for i, ts in enumerate(self._timestamps) if now - ts <= self._ttl]
        if len(keep_idx) != len(self._timestamps):
            self._stats["evictions_ttl"] += len(self._timestamps) - len(keep_idx)
            self._keys = [self._keys[i] for i in keep_idx]
            self._embeds = [self._embeds[i] for i in keep_idx]
            self._values = [self._values[i] for i in keep_idx]
            self._timestamps = [self._timestamps[i] for i in keep_idx]
            self._access = [self._access[i] for i in keep_idx]

    async def get(self, query: str) -> str | None:
        async with self._lock:
            self._evict_expired()
            # exact
            if query in self._keys:
                i = self._keys.index(query)
                self._access[i] = time.time()
                self._stats["exact_hits"] += 1
                return self._values[i]
            if not self._embeds:
                self._stats["misses"] += 1
                return None
            q_vec = self._embed(query)
            sims = [_cos(q_vec, e) for e in self._embeds]
            best_i = int(np.argmax(sims))
            if sims[best_i] >= self._threshold:
                self._access[best_i] = time.time()
                self._stats["semantic_hits"] += 1
                return self._values[best_i]
            self._stats["misses"] += 1
            return None

    async def set(self, query: str, value: str) -> None:
        async with self._lock:
            self._evict_expired()
            if len(self._keys) >= self._max_size:
                # LRU evict
                lru_i = int(np.argmin(self._access))
                self._stats["evictions_lru"] += 1
                self._keys.pop(lru_i); self._embeds.pop(lru_i)
                self._values.pop(lru_i); self._timestamps.pop(lru_i); self._access.pop(lru_i)
            now = time.time()
            self._keys.append(query)
            self._embeds.append(self._embed(query))
            self._values.append(value)
            self._timestamps.append(now)
            self._access.append(now)

    def stats(self) -> dict:
        total_hits = self._stats["exact_hits"] + self._stats["semantic_hits"]
        total = total_hits + self._stats["misses"]
        return {
            **self._stats,
            "total_queries": total,
            "size": len(self._keys),
            "hit_rate": total_hits / total if total else 0.0,
            "semantic_share": self._stats["semantic_hits"] / total_hits if total_hits else 0.0,
        }
