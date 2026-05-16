"""Shared preconditions for graph nodes."""
from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any

logger = logging.getLogger(__name__)


async def require_repo_cloned(state: dict, sandbox: Any) -> str | None:
    """Validate that repo is actually cloned and accessible.

    Returns an error message string if precondition fails, or None if OK.
    Cheap to call — does a single `test -d` in the sandbox.
    """
    if not state.get("repo_cloned"):
        return "repo not cloned (repo_cloned=False)"

    repo_dir = state.get("repo_dir")
    if not repo_dir:
        return "repo_dir missing from state"

    try:
        check = await asyncio.to_thread(
            sandbox.execute, f"test -d {shlex.quote(repo_dir)}/.git"
        )
        if check.exit_code != 0:
            return f"repo dir {repo_dir} no longer exists or is not a git repo"
    except Exception as exc:  # noqa: BLE001
        return f"sandbox check failed: {exc}"

    return None


async def get_files_changed(sandbox: Any, repo_dir: str) -> list[str]:
    """Return list of file paths changed in the working tree (not yet committed).

    Includes staged, unstaged, and untracked files.
    """
    safe_dir = shlex.quote(repo_dir)
    # --porcelain gives status; -uall includes untracked files
    result = await asyncio.to_thread(
        sandbox.execute,
        f"cd {safe_dir} && git status --porcelain -uall",
    )
    if result.exit_code != 0:
        logger.warning("git status failed: %s", result.output)
        return []

    files: list[str] = []
    for line in result.output.strip().splitlines():
        if len(line) < 4:
            continue
        # Format: "XY path" where XY is 2-char status
        path = line[3:].strip()
        # Handle rename: "old -> new" — take the new path
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        # Strip quotes that git adds for paths with special chars
        path = path.strip('"')
        if path:
            files.append(path)
    return files


async def get_diff_stat(sandbox: Any, repo_dir: str) -> str:
    """Return git diff --stat output for context in messages."""
    safe_dir = shlex.quote(repo_dir)
    result = await asyncio.to_thread(
        sandbox.execute,
        f"cd {safe_dir} && git diff --stat HEAD",
    )
    if result.exit_code != 0:
        return ""
    return result.output.strip()