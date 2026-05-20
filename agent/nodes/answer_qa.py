"""Answer QA node: read-only mini deep agent for questions."""

from __future__ import annotations

import logging
import os

from deepagents import create_deep_agent
from langchain_core.messages import AnyMessage
from langgraph.graph.state import RunnableConfig

from ..middleware import ToolErrorMiddleware
from ..tools import (
    fetch_url,
    get_pr_check_runs,
    get_pr_review_comments,
    http_request,
    web_search,
)
from ..utils.messages import extract_text_content
from ..utils.model import make_model
from ..utils.sandbox_state import get_sandbox_backend

logger = logging.getLogger(__name__)

DEFAULT_QA_MODEL_ID = os.environ.get("LLM_MODEL_ID", "nvidia:meta/llama-3.3-70b-instruct")
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
    branch_name = state.get("branch_name") or "<unknown>"

    repo = state.get("repo") or config.get("configurable", {}).get("repo") or {}
    pr_number = state.get("pr_number")
    github_issue = config.get("configurable", {}).get("github_issue") or {}

    github_context = f"""
Repository: {repo.get("owner")}/{repo.get("name")}
Branch: {branch_name}
Repo cloned at: {repo_dir}
"""
    if pr_number:
        github_context += f"PR Number: #{pr_number} (you can fetch PR review comments)\n"
    if github_issue.get("number"):
        github_context += (
            f"Issue Number: #{github_issue['number']} — {github_issue.get('title', '')}\n"
        )

    prompt = f"""You are answering a question about a software project.
You have FULL access to the codebase via bash commands — use them freely.

{github_context}

AGENTS.md: {agents_md}

Available capabilities:
- Run ANY bash command: grep, find, cat, git log, git blame, git diff, python, etc.
- Read any file in the repository
- Search git history: git log --oneline, git show, git blame
- Check PR review comments via get_pr_review_comments tool
- Check CI status via get_pr_check_runs tool
- Search the web via web_search for docs or context
- Fetch any URL via fetch_url

HOW TO ANSWER:
1. Explore the codebase using bash commands to gather facts
2. Check git history if the question is about why something changed
3. Read relevant source files, tests, configs
4. If question is about an issue/PR, fetch its comments for context
5. Give a thorough, accurate answer backed by what you found
6. Do NOT modify any files
7. Do NOT commit anything
"""

    agent = create_deep_agent(
        model=make_model(DEFAULT_QA_MODEL_ID, max_tokens=16_000),
        system_prompt=prompt,
        tools=[
            web_search,
            fetch_url,
            http_request,
            get_pr_review_comments,
            get_pr_check_runs,
        ],
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

    # Extract the final answer explicitly
    final_answer = ""
    for msg in reversed(inner_messages):
        if getattr(msg, "type", None) == "ai":
            from ..utils.messages import extract_text_content

            text = extract_text_content(msg.content)
            if text:
                final_answer = text
                break

    return {
        "messages": inner_messages[-5:],
        "qa_answer": final_answer,
    }
