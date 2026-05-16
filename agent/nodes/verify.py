"""Verify node: run formatters/linters on changed files only."""
from __future__ import annotations

import asyncio
import logging
import shlex

from langgraph.graph.state import RunnableConfig

from ..utils.preconditions import require_repo_cloned
from ..utils.sandbox_state import get_sandbox_backend

logger = logging.getLogger(__name__)


async def _run(sandbox, cmd: str, *, timeout: int = 120):
    return await asyncio.to_thread(sandbox.execute, cmd, timeout=timeout)


async def verify_node(state: dict, config: RunnableConfig) -> dict:
    """Run lint/format only on files that changed this run.

    If nothing changed, skip verification and advance the step counter.
    If linters are missing, skip gracefully — don't fail the task.
    """
    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    repo_dir = state.get("repo_dir")
    # Pre-increment: on first call this is 1, on second 2, etc.
    # route_after_verify checks `attempts >= MAX_VERIFICATION_ATTEMPTS` (default 3),
    # so we get exactly 3 tries before exhaustion.  On success we reset to 0.
    attempts = state.get("verification_attempts", 0) + 1
    current_step = state.get("current_step", 0)

    sandbox = await get_sandbox_backend(thread_id) if thread_id else None
    if not sandbox:
        return {
            "lint_passed": False,
            "tests_passed": False,
            "verification_attempts": attempts,
            "verification_error": "No sandbox available for verification",
        }

    precondition_error = await require_repo_cloned(state, sandbox)
    if precondition_error:
        return {
            "lint_passed": False,
            "tests_passed": False,
            "verification_attempts": attempts,
            "verification_error": f"verify precondition: {precondition_error}",
        }

    safe_dir = shlex.quote(repo_dir)
    files_changed = state.get("files_changed_so_far") or []

    # Nothing changed — agent may have done a read-only step; advance and continue
    if not files_changed:
        logger.info("No files changed; skipping verification, advancing step counter")
        return {
            "lint_passed": True,
            "tests_passed": True,
            "verification_attempts": 0,
            "verification_error": None,
            "current_step": current_step + 1,
        }

    has_python = any(f.endswith(".py") for f in files_changed)
    has_js_ts = any(f.endswith((".js", ".jsx", ".ts", ".tsx")) for f in files_changed)

    errors: list[str] = []

    # Python: try ruff directly first, then fall back to make lint
    if has_python:
        ruff_available = await _run(sandbox, "command -v ruff", timeout=10)
        if ruff_available.exit_code == 0:
            safe_files = " ".join(
                shlex.quote(f) for f in files_changed if f.endswith(".py")
            )
            result = await _run(
                sandbox,
                f"cd {safe_dir} && ruff check {safe_files}",
                timeout=60,
            )
            if result.exit_code != 0:
                errors.append(f"ruff check failed:\n{result.output}")
        else:
            # Fall back to Makefile
            makefile_exists = await _run(
                sandbox, f"test -f {safe_dir}/Makefile", timeout=5
            )
            if makefile_exists.exit_code == 0:
                result = await _run(
                    sandbox, f"cd {safe_dir} && make lint", timeout=120
                )
                if result.exit_code != 0:
                    errors.append(f"make lint failed:\n{result.output}")

    # JS/TS: yarn lint
    if has_js_ts:
        pkg_exists = await _run(
            sandbox, f"test -f {safe_dir}/package.json", timeout=5
        )
        if pkg_exists.exit_code == 0:
            result = await _run(
                sandbox, f"cd {safe_dir} && yarn lint", timeout=180
            )
            if result.exit_code != 0:
                errors.append(f"yarn lint failed:\n{result.output}")

    if errors:
        combined = "\n\n".join(errors)
        # Truncate so error doesn't blow the context window on retry
        if len(combined) > 6000:
            combined = combined[:6000] + "\n... [truncated]"
        logger.warning("Verification failed attempt %d", attempts)
        return {
            "lint_passed": False,
            "tests_passed": False,
            "verification_attempts": attempts,
            "verification_error": combined,
        }

    logger.info("Verification passed attempt %d", attempts)
    return {
        "lint_passed": True,
        "tests_passed": True,
        "verification_attempts": 0,
        "verification_error": None,
        "current_step": current_step + 1,
        "files_changed_so_far": [],
    }