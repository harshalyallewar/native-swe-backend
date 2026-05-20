"""Check queue node: drain pending messages from the LangGraph store."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from langchain_core.messages import HumanMessage
from langgraph.config import get_store
from langgraph.graph.state import RunnableConfig

from ..utils.multimodal import fetch_image_block

logger = logging.getLogger(__name__)

STOP_PATTERNS = re.compile(
    r"^\s*(stop|quit|exit|cancel|abort|halt|nevermind|never mind)\s*$", re.IGNORECASE
)


async def _classify_incoming_message(
    message_text: str,
    current_task_description: str,
    model,
) -> str:
    """Returns: 'stop' | 'related' | 'unrelated'"""
    from langchain_core.messages import HumanMessage as LCHumanMessage

    if STOP_PATTERNS.match(message_text.strip()):
        return "stop"

    prompt = f"""Current task being executed:
{current_task_description}

New incoming message:
{message_text}

Classify the new message as exactly one of:
- "stop": user wants to stop/cancel the current task
- "related": message adds to, modifies, or clarifies the current task
- "unrelated": message is about something completely different

Respond with only one word: stop, related, or unrelated"""

    try:
        result = model.invoke([LCHumanMessage(content=prompt)])
        content = result.content.strip().lower() if hasattr(result, "content") else "unrelated"
        if "stop" in content:
            return "stop"
        if "related" in content:
            return "related"
        return "unrelated"
    except Exception:
        return "unrelated"


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

    # Get text of the new message for classification
    new_message_text = " ".join(
        block.get("text", "")
        for block in content_blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )

    current_intent = state.get("intent")
    is_mid_task = current_intent is not None

    if is_mid_task and new_message_text:
        import os as _os

        from ..utils.model import make_model

        classify_model = make_model(
            _os.environ.get("CLASSIFY_MODEL_ID", _os.environ.get("LLM_MODEL_ID", "nvidia:llama-3.3-70b-instruct")), max_tokens=64
        )

        plan = state.get("plan") or []
        current_step = state.get("current_step", 0)
        task_desc = "\n".join(f"- {t['description']}" for t in plan) or "(no plan)"

        message_class = await _classify_incoming_message(
            new_message_text, str(task_desc), classify_model
        )

        if message_class == "stop":
            logger.info("Stop command received for thread %s", thread_id)
            return {
                "fatal_error": "Task stopped by user request.",
                "error_stage": "user_stop",
                "has_new_instructions": False,
            }

        if message_class == "unrelated":
            logger.info("Unrelated message received mid-task for thread %s", thread_id)
            ignore_msg = HumanMessage(
                content=(
                    f"[system] User sent an unrelated message while task is running: "
                    f'"{new_message_text[:100]}". '
                    f"Notify user that current task is still running."
                )
            )
            return {
                "messages": [ignore_msg],
                "has_new_instructions": False,
                "pending_user_notification": (
                    "I received your message but I'm currently working on another task. "
                    "I'll get to it when this one is complete, or send 'stop' to cancel."
                ),
            }

        if message_class == "related":
            logger.info(
                "Related message received mid-task for thread %s, appending to plan", thread_id
            )
            directive_msg = HumanMessage(
                content=(
                    f"[mid-task instruction] The following new instruction arrived "
                    f"while executing the current plan. "
                    f"DO NOT restart or discard already-completed steps. "
                    f"APPEND new todos at the end of the existing plan to honor this instruction. "
                    f"Already completed steps: {current_step} of {len(plan)}.\n\n"
                    f"New instruction: {new_message_text}"
                )
            )
            return {
                "messages": [directive_msg],
                "has_new_instructions": True,
            }

    # Original path: first message or post-task message
    new_message = HumanMessage(content=content_blocks)
    return {
        "messages": [new_message],
        "has_new_instructions": True,
    }
