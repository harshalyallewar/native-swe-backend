"""Entry node: validate config, resolve GitHub token, ensure sandbox exists."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.graph.state import RunnableConfig
from langgraph_sdk import get_client

from ..utils.auth import resolve_github_token
from ..utils.sandbox import create_sandbox
from ..utils.sandbox_state import SANDBOX_BACKENDS, get_sandbox_id_from_metadata

logger = logging.getLogger(__name__)

SANDBOX_CREATING = "__creating__"
SANDBOX_CREATION_TIMEOUT = 180
SANDBOX_POLL_INTERVAL = 1.0

_client = get_client()


async def _wait_for_sandbox_id(thread_id: str) -> str:
    elapsed = 0.0
    while elapsed < SANDBOX_CREATION_TIMEOUT:
        sandbox_id = await get_sandbox_id_from_metadata(thread_id)
        if sandbox_id is not None and sandbox_id != SANDBOX_CREATING:
            return sandbox_id
        await asyncio.sleep(SANDBOX_POLL_INTERVAL)
        elapsed += SANDBOX_POLL_INTERVAL
    raise TimeoutError(f"Timeout waiting for sandbox creation for thread {thread_id}")


async def _check_or_recreate_sandbox(sandbox_backend: Any, thread_id: str) -> Any:
    """Ping cached sandbox; recreate on failure."""
    try:
        await asyncio.to_thread(sandbox_backend.execute, "echo ok")
        return sandbox_backend
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Cached sandbox unreachable for thread %s, recreating: %s", thread_id, exc
        )
        SANDBOX_BACKENDS.pop(thread_id, None)
        await _client.threads.update(
            thread_id=thread_id,
            metadata={"sandbox_id": SANDBOX_CREATING},
        )
        try:
            new_backend = await asyncio.to_thread(create_sandbox)
        except Exception:
            logger.exception("Failed to recreate sandbox")
            await _client.threads.update(
                thread_id=thread_id, metadata={"sandbox_id": None}
            )
            raise
        return new_backend


async def entry_node(state: dict, config: RunnableConfig) -> dict:
    """Pure-Python setup: resolve GitHub token + ensure sandbox.

    Returns either {github_token, sandbox_id} or
    {fatal_error, error_stage: "entry"}.
    """
    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return {
            "fatal_error": "Missing thread_id in state/config",
            "error_stage": "entry",
        }

    # Resolve GitHub token
    try:
        github_token, encrypted = await resolve_github_token(config, thread_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to resolve GitHub token for thread %s", thread_id)
        return {
            "fatal_error": f"GitHub auth failed: {exc}",
            "error_stage": "entry",
        }

    # Persist encrypted token onto config metadata for downstream tools
    try:
        if "metadata" in config and isinstance(config["metadata"], dict):
            config["metadata"]["github_token_encrypted"] = encrypted
    except Exception:  # noqa: BLE001
        logger.debug("Could not stash encrypted token in config metadata")

    # Resolve / create sandbox
    sandbox_backend = SANDBOX_BACKENDS.get(thread_id)
    try:
        sandbox_id = await get_sandbox_id_from_metadata(thread_id)
    except Exception:  # noqa: BLE001
        sandbox_id = None

    try:
        if sandbox_id == SANDBOX_CREATING and not sandbox_backend:
            logger.info("Sandbox creation in progress, waiting...")
            sandbox_id = await _wait_for_sandbox_id(thread_id)

        if sandbox_backend:
            logger.info("Using cached sandbox backend for thread %s", thread_id)
            sandbox_backend = await _check_or_recreate_sandbox(sandbox_backend, thread_id)
        elif sandbox_id is None:
            logger.info("Creating new sandbox for thread %s", thread_id)
            await _client.threads.update(
                thread_id=thread_id, metadata={"sandbox_id": SANDBOX_CREATING}
            )
            try:
                sandbox_backend = await asyncio.to_thread(create_sandbox)
            except Exception:
                logger.exception("Failed to create sandbox")
                await _client.threads.update(
                    thread_id=thread_id, metadata={"sandbox_id": None}
                )
                raise
        else:
            logger.info("Connecting to existing sandbox %s", sandbox_id)
            try:
                sandbox_backend = await asyncio.to_thread(create_sandbox, sandbox_id)
            except Exception:
                logger.warning(
                    "Failed to connect to existing sandbox %s, creating new", sandbox_id
                )
                await _client.threads.update(
                    thread_id=thread_id, metadata={"sandbox_id": SANDBOX_CREATING}
                )
                try:
                    sandbox_backend = await asyncio.to_thread(create_sandbox)
                except Exception:
                    logger.exception("Failed to create replacement sandbox")
                    await _client.threads.update(
                        thread_id=thread_id, metadata={"sandbox_id": None}
                    )
                    raise
            sandbox_backend = await _check_or_recreate_sandbox(sandbox_backend, thread_id)
    except Exception as exc:  # noqa: BLE001
        return {
            "fatal_error": f"Sandbox setup failed: {exc}",
            "error_stage": "entry",
        }

    SANDBOX_BACKENDS[thread_id] = sandbox_backend

    # Persist sandbox_id + git identity config if we created a new sandbox
    try:
        if sandbox_id != sandbox_backend.id:
            await _client.threads.update(
                thread_id=thread_id,
                metadata={"sandbox_id": sandbox_backend.id},
            )
            await asyncio.to_thread(
                sandbox_backend.execute,
                "git config --global user.name 'native-swe[bot]' && "
                "git config --global user.email 'native-swe@users.noreply.github.com'",
            )
    except Exception:  # noqa: BLE001
        logger.debug("Failed to update sandbox metadata / git identity")

    return {
        "github_token": github_token,
        "sandbox_id": sandbox_backend.id,
    }
