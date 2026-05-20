"""Verify node: run formatters/linters on changed files only."""

from __future__ import annotations

import asyncio
import logging
import os
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
        logger.info("No files changed; skipping verification")
        return {
            "lint_passed": True,
            "tests_passed": True,
            "verification_attempts": 0,
            "verification_error": None,
            # No current_step increment — step management is now in execute_step
        }

    has_python = any(f.endswith(".py") for f in files_changed)
    has_js_ts = any(f.endswith((".js", ".jsx", ".ts", ".tsx")) for f in files_changed)

    errors: list[str] = []

    # --- Log verification_strategy from the plan for this step ---
    plan = state.get("plan") or []
    # current_step points to the step that just executed (not yet advanced)
    if 0 <= current_step < len(plan):
        strategy = plan[current_step].get("verification_strategy", "")
        if strategy and strategy.strip().lower() not in ("", "none", "lint"):
            logger.info("Step %d verification strategy from plan: %s", current_step, strategy)

    # Python: try ruff directly first, then fall back to make lint
    if has_python:
        ruff_available = await _run(sandbox, "command -v ruff", timeout=10)
        if ruff_available.exit_code == 0:
            safe_files = " ".join(shlex.quote(f) for f in files_changed if f.endswith(".py"))
            result = await _run(
                sandbox,
                f"cd {safe_dir} && ruff check {safe_files}",
                timeout=60,
            )
            if result.exit_code != 0:
                errors.append(f"ruff check failed:\n{result.output}")
        else:
            # Fall back to Makefile
            makefile_exists = await _run(sandbox, f"test -f {safe_dir}/Makefile", timeout=5)
            if makefile_exists.exit_code == 0:
                result = await _run(sandbox, f"cd {safe_dir} && make lint", timeout=120)
                if result.exit_code != 0:
                    errors.append(f"make lint failed:\n{result.output}")

    # JS/TS: yarn lint
    if has_js_ts:
        pkg_exists = await _run(sandbox, f"test -f {safe_dir}/package.json", timeout=5)
        if pkg_exists.exit_code == 0:
            result = await _run(sandbox, f"cd {safe_dir} && yarn lint", timeout=180)
            if result.exit_code != 0:
                errors.append(f"yarn lint failed:\n{result.output}")

    # --- Run tests for changed Python files ---
    if has_python:
        # Also run test files that were directly modified
        test_files_direct = [
            f
            for f in files_changed
            if f.endswith(".py")
            and (
                os.path.basename(f).startswith("test_") or os.path.basename(f).endswith("_test.py")
            )
        ]

        test_files: list[str] = []
        for f in files_changed:
            if not f.endswith(".py"):
                continue
            basename_no_ext = os.path.basename(f).replace(".py", "")
            # Skip files that are themselves tests
            if basename_no_ext.startswith("test_") or basename_no_ext.endswith("_test"):
                continue
            # Common patterns: test_foo.py, foo_test.py, tests/test_foo.py
            test_find = await _run(
                sandbox,
                f"find {safe_dir} -name 'test_{basename_no_ext}.py' "
                f"-o -name '{basename_no_ext}_test.py' 2>/dev/null | head -3",
                timeout=10,
            )
            if test_find.exit_code == 0 and test_find.output.strip():
                test_files.extend(test_find.output.strip().splitlines())

        all_test_files = test_files_direct + test_files
        if all_test_files:
            # Deduplicate and limit to 3 test files to keep runtime bounded
            unique_tests = list(dict.fromkeys(t.strip() for t in all_test_files))[:3]
            safe_test_files = " ".join(shlex.quote(t) for t in unique_tests)
            logger.info("Running related test files: %s", unique_tests)
            test_run = await _run(
                sandbox,
                f"cd {safe_dir} && python -m pytest {safe_test_files} -x -q "
                f"--no-header 2>&1 | tail -30",
                timeout=120,
            )
            if test_run.exit_code != 0:
                errors.append(f"Related tests failed:\n{test_run.output}")

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
        # current_step is NOT incremented here — execute_step does it
        # files_changed_so_far is NOT reset here — commit_pr does it
    }
