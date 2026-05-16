"""Execute step node: a narrow-scope mini deep agent for ONE todo item."""
from __future__ import annotations

import asyncio
import logging
import os
import shlex

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.graph.state import RunnableConfig

from ..middleware import SanitizeToolInputsMiddleware, ToolErrorMiddleware
from ..utils.model import make_model
from ..utils.preconditions import require_repo_cloned
from ..utils.sandbox_state import get_sandbox_backend

logger = logging.getLogger(__name__)

DEFAULT_EXECUTE_MODEL_ID = os.environ.get("LLM_MODEL_ID", "nvidia:meta/llama-3.1-70b-instruct")
EXECUTE_RECURSION_LIMIT = 40


async def _get_changed_files(sandbox, repo_dir: str) -> list[str]:
    """Return files changed in working tree (staged + unstaged + untracked)."""
    result = await asyncio.to_thread(
        sandbox.execute,
        f"cd {shlex.quote(repo_dir)} && git status --porcelain -uall",
    )
    if result.exit_code != 0:
        return []
    files = []
    for line in result.output.strip().splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip().strip('"')
        # Handle renames: "old -> new"
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            files.append(path)
    return files


async def execute_step_node(state: dict, config: RunnableConfig) -> dict:
    """Implement a single todo item using a narrow-scope deep agent."""
    plan = state.get("plan") or []
    current_step = state.get("current_step", 0)
    if current_step >= len(plan):
        logger.info(
            "execute_step called with current_step=%d but plan has %d steps; skipping",
            current_step,
            len(plan),
        )
        return {}

    todo = plan[current_step]
    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return {
            "fatal_error": "Missing thread_id",
            "error_stage": "execute_step",
        }

    sandbox = await get_sandbox_backend(thread_id)
    if not sandbox:
        return {
            "fatal_error": "No sandbox available for execute_step",
            "error_stage": "execute_step",
        }

    precondition_error = await require_repo_cloned(state, sandbox)
    if precondition_error:
        return {
            "fatal_error": f"execute_step precondition: {precondition_error}",
            "error_stage": "execute_step",
        }

    repo_dir = state.get("repo_dir")
    branch_name = state.get("branch_name")
    agents_md = state.get("agents_md_content") or "No AGENTS.md present."

    verification_error = state.get("verification_error")
    attempts = state.get("verification_attempts", 0)
    retry_context = ""
    if verification_error and attempts > 0:
        retry_context = (
            f"\n\nNOTE: A previous verification attempt FAILED with this error "
            f"(attempt {attempts}). Fix it before declaring done:\n"
            f"```\n{verification_error}\n```\n"
        )

    files_hint = todo.get("files_likely_touched") or []
    files_str = ", ".join(files_hint) if files_hint else "(unknown — discover them)"

    prompt = f"""You are implementing one specific task step in an already-cloned repository.

Repository location: {repo_dir}
Branch: {branch_name}

AGENTS.md rules (MANDATORY — follow these exactly):
{agents_md}

Your ONLY job right now:
{todo["description"]}

Files likely involved: {files_str}
{retry_context}
Rules:
- Do NOT run linters or formatters. That happens automatically after you finish.
- Do NOT run tests. That happens automatically after you finish.
- Do NOT commit anything. That happens automatically after you finish.
- Do NOT message anyone. That happens automatically after you finish.
- Do NOT move on to other todo items. Only do the one task above.
- When you are done, stop calling tools.
"""

    agent = create_deep_agent(
        model=make_model(DEFAULT_EXECUTE_MODEL_ID, max_tokens=16_000),
        system_prompt=prompt,
        tools=[],
        backend=sandbox,
        middleware=[
            ToolErrorMiddleware(),
            SanitizeToolInputsMiddleware(),
        ],
    )

    invoke_config = {**config, "recursion_limit": EXECUTE_RECURSION_LIMIT}

    try:
        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": todo["description"]}
                ]
            },
            config=invoke_config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("execute_step inner agent failed")
        return {
            "fatal_error": f"execute_step failed: {exc}",
            "error_stage": "execute_step",
        }

    inner_messages = result.get("messages", []) if isinstance(result, dict) else []
    tail = inner_messages[-3:] if inner_messages else []

    # Track which files actually changed
    files_changed = await _get_changed_files(sandbox, repo_dir)
    accumulated = list(state.get("files_changed_so_far") or [])
    for f in files_changed:
        if f not in accumulated:
            accumulated.append(f)

    logger.info(
        "execute_step completed step %d/%d — changed files: %s",
        current_step + 1,
        len(plan),
        ", ".join(files_changed) or "(none)",
    )

    summary_msg = HumanMessage(
        content=(
            f"[execute_step #{current_step + 1}] "
            f"Changed {len(files_changed)} file(s): "
            f"{', '.join(files_changed) or '(none)'}"
        )
    )

    return {
        "messages": [*tail, summary_msg],
        "files_changed_so_far": accumulated,
    }