"""Classify node: single LLM call to classify task intent."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.state import RunnableConfig
from pydantic import BaseModel, Field

from ..utils.messages import extract_text_content
from ..utils.model import make_model

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFY_MODEL_ID = os.environ.get(
    "CLASSIFY_MODEL_ID", os.environ.get("LLM_MODEL_ID", "nvidia:meta/llama-3.3-70b-instruct")
)


class ClassifyOutput(BaseModel):
    intent: Literal["code_change", "question", "review"] = Field(
        description="The kind of task the user is asking for."
    )
    reasoning: str = Field(
        description="One sentence explaining why this intent was chosen.",
        default="",
    )


def _last_human_text(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            text = extract_text_content(msg.content) or ""
            # Skip injected synthetic messages from nodes
            if text and not text.startswith("["):
                return text
    return ""


async def classify_node(state: dict, config: RunnableConfig) -> dict:
    """Classify the user's intent into one of three buckets."""
    repo = state.get("repo") or {}
    owner = repo.get("owner", "unknown")
    name = repo.get("name", "unknown")

    messages = state.get("messages") or []
    last_user_text = _last_human_text(messages) or "(no user text)"

    agents_present = "yes" if state.get("agents_md_content") else "no"

    prompt = f"""Given this task and conversation, classify the intent.

code_change: requires modifying files in the repository
question: read-only, answering questions about code or repo
review: addressing PR review comments on an existing PR

Repository: {owner}/{name}
AGENTS.md present: {agents_present}

Task: {last_user_text}

Respond with the intent and a one-sentence reasoning."""

    try:

        def _classify_sync():
            model = make_model(DEFAULT_CLASSIFY_MODEL_ID, max_tokens=512)
            structured = model.with_structured_output(ClassifyOutput)
            return structured.invoke([HumanMessage(content=prompt)])

        result: ClassifyOutput = await asyncio.to_thread(_classify_sync)
    except Exception as exc:  # noqa: BLE001
        logger.exception("classify_node failed; defaulting to 'question'")
        return {
            "intent": "question",
            "messages": [
                HumanMessage(
                    content=(
                        f"[classify_node fallback] Could not classify intent ({exc}); "
                        f"treating as a question."
                    )
                )
            ],
        }

    logger.info("Classified intent=%s reasoning=%s", result.intent, result.reasoning)
    return {"intent": result.intent}
