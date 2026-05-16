"""Plan node: single LLM call to produce a structured plan for code_change tasks."""
from __future__ import annotations

import asyncio
import logging
import os
import shlex

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.state import RunnableConfig
from pydantic import BaseModel, Field

from ..utils.messages import extract_text_content
from ..utils.model import make_model
from ..utils.sandbox_state import get_sandbox_backend

logger = logging.getLogger(__name__)

DEFAULT_PLAN_MODEL_ID = os.environ.get(
    "PLAN_MODEL_ID", os.environ.get("LLM_MODEL_ID", "nvidia:meta/llama-3.1-70b-instruct")
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
    todos: list[PlanTodo] = Field(
        description=f"Ordered list of steps (max {MAX_PLAN_STEPS})."
    )
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


async def plan_node(state: dict, config: RunnableConfig) -> dict:
    """Produce a structured plan. Peeks at the repo file tree first so
    files_likely_touched contains real paths instead of hallucinated ones."""
    repo_dir = state.get("repo_dir") or "<unknown>"
    branch_name = state.get("branch_name") or "<unknown>"
    agents_md = state.get("agents_md_content") or "none"

    messages = state.get("messages") or []
    user_message = _last_human_text(messages) or "(no user text)"

    # Get real file tree from sandbox so LLM uses actual paths
    file_tree = "(repo not available)"
    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    if thread_id and state.get("repo_cloned"):
        try:
            sandbox = await get_sandbox_backend(thread_id)
            if sandbox:
                file_tree = await _get_file_tree(sandbox, repo_dir)
        except Exception:  # noqa: BLE001
            logger.warning("Could not fetch file tree for planning", exc_info=True)

    prompt = f"""You are planning a code change task. Produce a step-by-step plan.
Each step should be independently completable.
Keep steps small — one logical change per step.
Maximum {MAX_PLAN_STEPS} steps. If simpler, use fewer.

Repository: {repo_dir}
Branch: {branch_name}
AGENTS.md rules: {agents_md}

Files in this repository (use these EXACT paths in files_likely_touched):
{file_tree}

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

    todos = [
        {
            "description": t.description,
            "files_likely_touched": t.files_likely_touched,
            "verification_strategy": t.verification_strategy,
            "completed": False,
        }
        for t in result.todos[:MAX_PLAN_STEPS]
    ]

    if not todos:
        return {
            "fatal_error": "Planning produced zero steps",
            "error_stage": "plan",
        }

    logger.info(
        "plan_node produced %d step(s); summary=%s", len(todos), result.summary
    )

    summary_message = HumanMessage(
        content=f"[plan] {result.summary or 'Plan generated'} ({len(todos)} step(s))."
    )

    return {
        "plan": todos,
        "current_step": 0,
        "verification_attempts": 0,
        "verification_error": None,
        "lint_passed": None,
        "tests_passed": None,
        "files_changed_so_far": [],
        "messages": [summary_message],
    }