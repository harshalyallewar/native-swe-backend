"""Check queue node: drain pending messages from the LangGraph store."""
from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.messages import HumanMessage
from langgraph.config import get_store
from langgraph.graph.state import RunnableConfig

from ..utils.multimodal import fetch_image_block

logger = logging.getLogger(__name__)


async def _build_blocks_from_payload(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    text = payload.get("text", "")
    image_urls = payload.get("image_urls", []) or []
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})

    if not image_urls:
        return blocks
    async with httpx.AsyncClient() as client:
        for image_url in image_urls:
            image_block = await fetch_image_block(image_url, client)
            if image_block:
                blocks.append(image_block)
    return blocks


async def check_queue_node(state: dict, config: RunnableConfig) -> dict:
    """Pull pending messages from the store and inject as new HumanMessage.

    Returns has_new_instructions=True if messages were injected,
    False otherwise.
    """
    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return {"has_new_instructions": False}

    try:
        store = get_store()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not get store: %s", exc)
        return {"has_new_instructions": False}

    if store is None:
        return {"has_new_instructions": False}

    namespace = ("queue", thread_id)
    try:
        queued_item = await store.aget(namespace, "pending_messages")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to get queued item: %s", exc)
        return {"has_new_instructions": False}

    if queued_item is None:
        return {"has_new_instructions": False}

    queued_value = queued_item.value
    queued_messages = queued_value.get("messages", [])

    # Delete IMMEDIATELY (before processing) to prevent duplicate injection.
    try:
        await store.adelete(namespace, "pending_messages")
    except Exception:  # noqa: BLE001
        logger.warning("Failed to delete queued messages", exc_info=True)

    if not queued_messages:
        return {"has_new_instructions": False}

    logger.info(
        "Found %d queued message(s) for thread %s; injecting into state",
        len(queued_messages),
        thread_id,
    )

    content_blocks: list[dict[str, Any]] = []
    for msg in queued_messages:
        content = msg.get("content")
        if isinstance(content, dict) and ("text" in content or "image_urls" in content):
            blocks = await _build_blocks_from_payload(content)
            content_blocks.extend(blocks)
        elif isinstance(content, list):
            content_blocks.extend(content)
        elif isinstance(content, str) and content:
            content_blocks.append({"type": "text", "text": content})

    if not content_blocks:
        return {"has_new_instructions": False}

    new_message = HumanMessage(content=content_blocks)

    return {
        "messages": [new_message],
        "has_new_instructions": True,
    }
