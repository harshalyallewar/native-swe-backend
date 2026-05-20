"""Plan node: single LLM call to produce a structured plan for code_change tasks."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.state import RunnableConfig
from pydantic import BaseModel, Field

from ..utils.messages import extract_text_content
from ..utils.model import make_model
from ..utils.sandbox_state import get_sandbox_backend

logger = logging.getLogger(__name__)

DEFAULT_PLAN_MODEL_ID = os.environ.get(
    "PLAN_MODEL_ID", os.environ.get("LLM_MODEL_ID", "nvidia:meta/llama-3.3-70b-instruct")
)

MAX_PLAN_STEPS = 5


class PlanTodo(BaseModel):
    description: str = Field(description="What this step accomplishes.")
    files_likely_touched: list[str] = Field(
        default_factory=list,
        description="Files that will probably be modified in this step.",
    )
    verification_strategy: str = Field(
        default="",
        description="How to verify this step worked (lint, tests, manual check).",
    )


class PlanOutput(BaseModel):
    todos: list[PlanTodo] = Field(description=f"Ordered list of steps (max {MAX_PLAN_STEPS}).")
    summary: str = Field(
        default="",
        description="One-sentence summary of the overall plan.",
    )


def _last_human_text(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            text = extract_text_content(msg.content) or ""
            # Skip injected synthetic messages from nodes
            if text and not text.startswith("["):
                return text
    return ""


async def _get_file_tree(sandbox, repo_dir: str) -> str:
    """Get a list of source files in the repo so the planner sees real paths."""
    result = await asyncio.to_thread(
        sandbox.execute,
        f"cd {shlex.quote(repo_dir)} && find . -type f "
        f"\\( -name '*.py' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' "
        f"-o -name '*.jsx' -o -name '*.go' -o -name '*.rs' -o -name '*.java' \\) "
        f"-not -path '*/node_modules/*' -not -path '*/.git/*' "
        f"-not -path '*/__pycache__/*' -not -path '*/dist/*' -not -path '*/build/*' "
        f"| sort | head -150",
    )
    if result.exit_code != 0 or not result.output.strip():
        return "(could not read file tree)"
    return result.output.strip()


async def _get_relevant_file_contents(sandbox, repo_dir: str, task_text: str) -> str:
    """Read a small amount of key file content to ground the plan.

    Reads README / AGENTS docs and any source files whose names or contents
    match identifiers extracted from the task description.
    """
    safe_dir = shlex.quote(repo_dir)
    contents: list[str] = []

    # --- 1. Read README.md / AGENTS.md ---
    for readme in ["README.md", "AGENTS.md"]:
        try:
            result = await asyncio.to_thread(
                sandbox.execute,
                f"test -f {safe_dir}/{readme} && head -80 {safe_dir}/{readme}",
            )
            if result.exit_code == 0 and result.output.strip():
                contents.append(f"=== {readme} (first 80 lines) ===\n{result.output.strip()}")
        except Exception:  # noqa: BLE001
            logger.debug("Could not read %s", readme, exc_info=True)

    # --- 2. Grep for identifiers mentioned in the task description ---
    identifiers = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b", task_text)
    # Deduplicate while preserving order, take first 5 unique
    seen: set[str] = set()
    unique_ids: list[str] = []
    for ident in identifiers:
        low = ident.lower()
        if low not in seen:
            seen.add(low)
            unique_ids.append(ident)
        if len(unique_ids) >= 5:
            break

    if unique_ids:
        identifier_pattern = "|".join(unique_ids)
        try:
            grep_result = await asyncio.to_thread(
                sandbox.execute,
                f"cd {safe_dir} && grep -rEl --include='*.py' --include='*.ts' "
                f"--include='*.tsx' --include='*.js' --include='*.go' "
                f"--include='*.rs' --include='*.java' "
                f"{shlex.quote(identifier_pattern)} . 2>/dev/null | head -5",
            )
            if grep_result.exit_code == 0 and grep_result.output.strip():
                relevant_files = grep_result.output.strip().splitlines()[:3]
                for rel_file in relevant_files:
                    clean = rel_file.lstrip("./")
                    try:
                        read_result = await asyncio.to_thread(
                            sandbox.execute,
                            f"head -60 {safe_dir}/{shlex.quote(clean)}",
                        )
                        if read_result.exit_code == 0 and read_result.output.strip():
                            contents.append(
                                f"=== {clean} (first 60 lines) ===\n{read_result.output.strip()}"
                            )
                    except Exception:  # noqa: BLE001
                        logger.debug("Could not read %s", clean, exc_info=True)
        except Exception:  # noqa: BLE001
            logger.debug("Grep for identifiers failed", exc_info=True)

    return "\n\n".join(contents) if contents else "(no key files read)"


async def plan_node(state: dict, config: RunnableConfig) -> dict:
    """Produce a structured plan. Peeks at the repo file tree first so
    files_likely_touched contains real paths instead of hallucinated ones."""
    repo_dir = state.get("repo_dir") or "<unknown>"
    branch_name = state.get("branch_name") or "<unknown>"
    agents_md = state.get("agents_md_content") or "none"

    messages = state.get("messages") or []
    user_message = _last_human_text(messages) or "(no user text)"

    # Detect mid-task append mode
    is_append_mode = "[mid-task instruction]" in user_message
    existing_plan = state.get("plan") or []
    completed_steps = [t for t in existing_plan if t.get("completed")]
    pending_steps = [t for t in existing_plan if not t.get("completed")]
    current_step = state.get("current_step", 0)

    if is_append_mode:
        append_context = f"""
IMPORTANT: This is an APPEND operation. Do NOT recreate or modify these already-completed steps:
{chr(10).join(f"  DONE - {t["description"]}" for t in completed_steps)}

These steps are still pending and must be kept as-is:
{chr(10).join(f"  PENDING - {t["description"]}" for t in pending_steps)}

Generate ONLY the new steps needed to honor the new instruction.
New steps must not redo what completed steps already did.
Max {MAX_PLAN_STEPS - len(pending_steps)} new steps.
"""
    else:
        append_context = ""

    # Get real file tree from sandbox so LLM uses actual paths
    file_tree = "(repo not available)"
    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    relevant_contents = "(repo not available)"
    if thread_id and state.get("repo_cloned"):
        try:
            sandbox = await get_sandbox_backend(thread_id)
            if sandbox:
                file_tree = await _get_file_tree(sandbox, repo_dir)
                relevant_contents = await _get_relevant_file_contents(
                    sandbox, repo_dir, user_message
                )
        except Exception:  # noqa: BLE001
            logger.warning("Could not fetch repo context for planning", exc_info=True)

    prompt = f"""You are planning a code change task. Produce a step-by-step plan.
Each step should be independently completable.
Keep steps small — one logical change per step.
Maximum {MAX_PLAN_STEPS} steps. If simpler, use fewer.

Repository: {repo_dir}
Branch: {branch_name}
AGENTS.md rules: {agents_md}
{append_context}
Files in this repository (use these EXACT paths in files_likely_touched):
{file_tree}

Key file contents for context:
{relevant_contents}

Task: {user_message}

Produce a structured plan with todos and a brief summary.
Only reference files that are actually listed above."""

    try:

        def _plan_sync():
            model = make_model(DEFAULT_PLAN_MODEL_ID, max_tokens=2048)
            structured = model.with_structured_output(PlanOutput)
            return structured.invoke([HumanMessage(content=prompt)])

        result: PlanOutput = await asyncio.to_thread(_plan_sync)
    except Exception as exc:  # noqa: BLE001
        logger.exception("plan_node failed")
        return {
            "fatal_error": f"Planning failed: {exc}",
            "error_stage": "plan",
        }

    new_todos = [
        {
            "description": t.description,
            "files_likely_touched": t.files_likely_touched,
            "verification_strategy": t.verification_strategy,
            "completed": False,
            "files_actually_changed": [],
            "outcome_summary": "",
        }
        for t in result.todos[:MAX_PLAN_STEPS]
    ]

    if not new_todos:
        return {
            "fatal_error": "Planning produced zero steps",
            "error_stage": "plan",
        }

    if is_append_mode:
        merged_plan = existing_plan + new_todos
        logger.info(
            "Appended %d new step(s) to existing plan. Total: %d", len(new_todos), len(merged_plan)
        )
        summary_message = HumanMessage(
            content=(
                f"[plan] Appended {len(new_todos)} new step(s) to honor new instruction. "
                f"Resuming from step {current_step + 1}."
            )
        )
        return {
            "plan": merged_plan,
            "verification_attempts": 0,
            "verification_error": None,
            "lint_passed": None,
            "tests_passed": None,
            "messages": [summary_message],
            # DO NOT reset current_step or files_changed_so_far in append mode
        }

    # Normal new-task mode
    summary_message = HumanMessage(
        content=f"[plan] {result.summary or 'Plan generated'} ({len(new_todos)} step(s))."
    )
    return {
        "plan": new_todos,
        "current_step": 0,
        "verification_attempts": 0,
        "verification_error": None,
        "lint_passed": None,
        "tests_passed": None,
        "files_changed_so_far": [],
        "messages": [summary_message],
    }
