"""Review PR node: mini deep agent for addressing PR review comments."""

from __future__ import annotations

import logging
import os

from deepagents import create_deep_agent
from langchain_core.messages import AnyMessage
from langgraph.graph.state import RunnableConfig

from ..middleware import SanitizeToolInputsMiddleware, ToolErrorMiddleware
from ..tools import (
    get_pr_check_runs,
    get_pr_review_comments,
)
from ..utils.messages import extract_text_content
from ..utils.model import make_model
from ..utils.preconditions import require_repo_cloned
from ..utils.sandbox_state import get_sandbox_backend

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_MODEL_ID = os.environ.get("LLM_MODEL_ID", "nvidia:meta/llama-3.3-70b-instruct")
REVIEW_RECURSION_LIMIT = 40


def _last_human_text(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            text = extract_text_content(msg.content) or ""
            # Skip injected synthetic messages from nodes
            if text and not text.startswith("["):
                return text
    return ""


async def review_pr_node(state: dict, config: RunnableConfig) -> dict:
    """Address PR review comments by making changes. Verify+commit handle the rest."""
    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return {
            "fatal_error": "Missing thread_id for review_pr",
            "error_stage": "review_pr",
        }

    sandbox = await get_sandbox_backend(thread_id)
    if not sandbox:
        return {
            "fatal_error": "No sandbox available for review_pr",
            "error_stage": "review_pr",
        }

    precondition_error = await require_repo_cloned(state, sandbox)
    if precondition_error:
        return {
            "fatal_error": f"review_pr precondition: {precondition_error}",
            "error_stage": "review_pr",
        }

    repo_dir = state.get("repo_dir")
    branch_name = state.get("branch_name")
    agents_md = state.get("agents_md_content") or "none"

    messages = state.get("messages") or []
    user_message = _last_human_text(messages) or (
        "Address the PR review comments on the current pull request."
    )

    verification_error = state.get("verification_error")
    attempts = state.get("verification_attempts", 0)
    retry_context = ""
    if verification_error and attempts > 0:
        retry_context = (
            f"\n\nNOTE: A previous verification attempt FAILED with this error "
            f"(attempt {attempts}). Fix it before declaring done:\n"
            f"```\n{verification_error}\n```\n"
        )

    prompt = f"""You are addressing PR review comments on an existing pull request.
The repository is already cloned at: {repo_dir}
Branch: {branch_name}

AGENTS.md: {agents_md}

Fetch the review comments, understand what changes are needed,
and implement them.{retry_context}
Do NOT commit. Do NOT notify anyone. Just make the code changes.
When done, stop calling tools.
"""

    agent = create_deep_agent(
        model=make_model(DEFAULT_REVIEW_MODEL_ID, max_tokens=16_000),
        system_prompt=prompt,
        tools=[
            get_pr_review_comments,
            get_pr_check_runs,
        ],
        backend=sandbox,
        middleware=[ToolErrorMiddleware(), SanitizeToolInputsMiddleware()],
    )

    invoke_config = {**config, "recursion_limit": REVIEW_RECURSION_LIMIT}

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=invoke_config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("review_pr inner agent failed")
        return {
            "fatal_error": f"review_pr failed: {exc}",
            "error_stage": "review_pr",
        }

    inner_messages = result.get("messages", []) if isinstance(result, dict) else []
    return {"messages": inner_messages[-5:]}
