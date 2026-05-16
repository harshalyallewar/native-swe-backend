"""Answer QA node: read-only mini deep agent for questions."""
from __future__ import annotations

import logging
import os

from deepagents import create_deep_agent
from langchain_core.messages import AnyMessage
from langgraph.graph.state import RunnableConfig

from ..middleware import ToolErrorMiddleware
from ..tools import fetch_url, http_request, web_search
from ..utils.messages import extract_text_content
from ..utils.model import make_model
from ..utils.sandbox_state import get_sandbox_backend

logger = logging.getLogger(__name__)

DEFAULT_QA_MODEL_ID = os.environ.get("LLM_MODEL_ID", "nvidia:meta/llama-3.1-70b-instruct")
QA_RECURSION_LIMIT = 30


def _last_human_text(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            text = extract_text_content(msg.content) or ""
            # Skip injected synthetic messages from nodes
            if text and not text.startswith("["):
                return text
    return ""


async def answer_qa_node(state: dict, config: RunnableConfig) -> dict:
    """Answer a read-only question about the repo."""
    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return {
            "fatal_error": "Missing thread_id for answer_qa",
            "error_stage": "answer_qa",
        }

    sandbox = await get_sandbox_backend(thread_id)
    if not sandbox:
        return {
            "fatal_error": "No sandbox available for answer_qa",
            "error_stage": "answer_qa",
        }

    messages = state.get("messages") or []
    user_message = _last_human_text(messages) or "Please answer the user's question."
    repo_dir = state.get("repo_dir") or "<unknown>"
    agents_md = state.get("agents_md_content") or "none"

    prompt = f"""You are answering a question about a code repository.
The repository is already cloned at: {repo_dir}

AGENTS.md: {agents_md}

Answer the question thoroughly using the available tools.
Do NOT modify any files.
Do NOT commit anything.
When you have a complete answer, stop calling tools.
"""

    agent = create_deep_agent(
        model=make_model(DEFAULT_QA_MODEL_ID, max_tokens=16_000),
        system_prompt=prompt,
        tools=[web_search, fetch_url, http_request],
        backend=sandbox,
        middleware=[ToolErrorMiddleware()],
    )

    invoke_config = {**config, "recursion_limit": QA_RECURSION_LIMIT}

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=invoke_config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("answer_qa inner agent failed")
        return {
            "fatal_error": f"answer_qa failed: {exc}",
            "error_stage": "answer_qa",
        }

    inner_messages = result.get("messages", []) if isinstance(result, dict) else []
    return {"messages": inner_messages[-5:]}
