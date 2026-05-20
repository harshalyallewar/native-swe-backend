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
from ..tools import fetch_url, http_request, web_search
from ..utils.model import make_model
from ..utils.preconditions import require_repo_cloned
from ..utils.sandbox_state import get_sandbox_backend

logger = logging.getLogger(__name__)

DEFAULT_EXECUTE_MODEL_ID = os.environ.get("LLM_MODEL_ID", "nvidia:meta/llama-3.3-70b-instruct")
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
    files_changed_so_far = list(state.get("files_changed_so_far") or [])

    # Build detailed previous steps context
    prev_steps_context = ""
    if current_step > 0:
        prev_steps = plan[:current_step]
        prev_lines = ["Previous completed steps and their outcomes:"]
        for i, prev in enumerate(prev_steps):
            status = "COMPLETED" if prev.get("completed") else "DONE"
            prev_lines.append(f"\n  Step {i + 1} [{status}]: {prev['description']}")
            if prev.get("files_actually_changed"):
                prev_lines.append(f"  Files changed: {', '.join(prev['files_actually_changed'])}")
            if prev.get("outcome_summary"):
                prev_lines.append(f"  Outcome: {prev['outcome_summary']}")

        if files_changed_so_far:
            prev_lines.append(
                f"\nAll files modified so far in this task: {', '.join(files_changed_so_far)}"
            )
        prev_lines.append(
            "\nIMPORTANT: Read the files listed above before writing any code "
            "that depends on them. Their contents reflect ALL changes from previous steps."
        )
        prev_steps_context = "\n".join(prev_lines)

    remaining_steps = plan[current_step + 1 :]
    remaining_context = ""
    if remaining_steps:
        remaining_lines = [
            "\nUpcoming steps after this one (for awareness only, do NOT implement them now):"
        ]
        for i, upcoming in enumerate(remaining_steps):
            remaining_lines.append(f"  Step {current_step + i + 2}: {upcoming['description']}")
        remaining_context = "\n".join(remaining_lines)

    prompt = f"""You are a software engineer implementing one specific step of a larger task.
The repository is already cloned at: {repo_dir}
Branch: {branch_name}

AGENTS.md rules (MANDATORY — follow these exactly):
{agents_md}

{prev_steps_context}

YOUR CURRENT TASK (Step {current_step + 1} of {len(plan)}):
{todo["description"]}

Files likely involved: {files_str}
{remaining_context}
{retry_context}

HOW TO WORK:
1. READ first — before writing any code, read all relevant files.
   If previous steps changed files you depend on, read their CURRENT content.
2. IMPLEMENT the change. Write clean minimal code matching existing style.
3. VERIFY your own work — re-read changed files to confirm correctness.
4. USE bash freely — you can run any command: grep, find, cat, python, npm, etc.
5. USE web_search or fetch_url if you need docs, examples, or API references.
6. STOP when this step is done. Do NOT implement other steps.
   Do NOT run linters. Do NOT commit. Those happen automatically.

You have full bash access. Use it. Read before you write.
"""

    agent = create_deep_agent(
        model=make_model(DEFAULT_EXECUTE_MODEL_ID, max_tokens=16_000),
        system_prompt=prompt,
        tools=[web_search, fetch_url, http_request],
        backend=sandbox,
        middleware=[
            ToolErrorMiddleware(),
            SanitizeToolInputsMiddleware(),
        ],
    )

    invoke_config = {**config, "recursion_limit": EXECUTE_RECURSION_LIMIT}

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": todo["description"]}]},
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

    # Update plan with per-step file tracking
    updated_plan = list(plan)
    updated_plan[current_step] = {
        **todo,
        "completed": True,
        "files_actually_changed": files_changed,
        "outcome_summary": f"Changed {len(files_changed)} file(s): {', '.join(files_changed) or 'none'}",
    }

    summary_msg = HumanMessage(
        content=(
            f"[execute_step #{current_step + 1}/{len(plan)}] "
            f"Completed: {todo['description'][:80]}. "
            f"Files changed: {', '.join(files_changed) or '(none)'}"
        )
    )

    return {
        "messages": [*tail, summary_msg],
        "files_changed_so_far": accumulated,
        "current_step": current_step + 1,  # advance step counter HERE not in verify
        "plan": updated_plan,
    }
