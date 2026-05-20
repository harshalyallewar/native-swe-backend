"""Shared sandbox state used by server and middleware."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any

from langgraph.config import get_config

from .sandbox import create_sandbox

logger = logging.getLogger(__name__)

# Max sandboxes to keep in memory at once (LRU eviction after this)
_MAX_SANDBOX_BACKENDS = 50

# Evict sandboxes not accessed for more than 2 hours
_SANDBOX_TTL_SECONDS = 7200

# Thread ID -> SandboxBackend (OrderedDict for LRU ordering)
SANDBOX_BACKENDS: OrderedDict[str, Any] = OrderedDict()

# Thread ID -> last access timestamp
_SANDBOX_LAST_ACCESS: dict[str, float] = {}


def _touch(thread_id: str) -> None:
    """Update last-access time and move to end (most recently used)."""
    _SANDBOX_LAST_ACCESS[thread_id] = time.monotonic()
    if thread_id in SANDBOX_BACKENDS:
        SANDBOX_BACKENDS.move_to_end(thread_id)


def evict_sandbox_backend(thread_id: str) -> None:
    """Explicitly remove a sandbox from the cache (call on thread completion)."""
    removed = SANDBOX_BACKENDS.pop(thread_id, None)
    _SANDBOX_LAST_ACCESS.pop(thread_id, None)
    if removed is not None:
        logger.info("Evicted sandbox backend for thread %s", thread_id)


def evict_stale_sandbox_backends() -> None:
    """Remove sandboxes idle longer than _SANDBOX_TTL_SECONDS, then enforce max size."""
    now = time.monotonic()

    # TTL eviction
    stale = [
        tid for tid, last in _SANDBOX_LAST_ACCESS.items()
        if now - last > _SANDBOX_TTL_SECONDS
    ]
    for tid in stale:
        evict_sandbox_backend(tid)
    if stale:
        logger.info("TTL-evicted %d stale sandbox(es)", len(stale))

    # LRU cap: evict oldest entries if over the limit
    while len(SANDBOX_BACKENDS) > _MAX_SANDBOX_BACKENDS:
        oldest_tid, _ = next(iter(SANDBOX_BACKENDS.items()))
        evict_sandbox_backend(oldest_tid)
        logger.warning(
            "LRU cap reached (%d), evicted oldest sandbox: %s",
            _MAX_SANDBOX_BACKENDS,
            oldest_tid,
        )


async def get_sandbox_id_from_metadata(thread_id: str) -> str | None:
    """Fetch sandbox_id from thread metadata."""
    try:
        config = get_config()
    except Exception:
        logger.exception("Failed to read thread metadata for sandbox")
        return None
    return config.get("metadata", {}).get("sandbox_id")


async def get_sandbox_backend(thread_id: str) -> Any | None:
    """Get sandbox backend from cache, or connect using thread metadata."""
    # Evict stale entries on every access (cheap check)
    evict_stale_sandbox_backends()

    sandbox_backend = SANDBOX_BACKENDS.get(thread_id)
    if sandbox_backend:
        _touch(thread_id)
        return sandbox_backend

    sandbox_id = await get_sandbox_id_from_metadata(thread_id)
    if not sandbox_id:
        raise ValueError(f"Missing sandbox_id in thread metadata for {thread_id}")

    sandbox_backend = await asyncio.to_thread(create_sandbox, sandbox_id)

    SANDBOX_BACKENDS[thread_id] = sandbox_backend
    _touch(thread_id)
    return sandbox_backend


def get_sandbox_backend_sync(thread_id: str) -> Any | None:
    """Sync wrapper for get_sandbox_backend."""
    return asyncio.run(get_sandbox_backend(thread_id))