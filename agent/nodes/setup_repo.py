"""Setup repo node: clone repo, checkout branch, read AGENTS.md. Pure Python."""
from __future__ import annotations

import asyncio
import logging
import shlex

from langgraph.graph.state import RunnableConfig

from ..utils.github import get_github_default_branch
from ..utils.sandbox_paths import aresolve_sandbox_work_dir
from ..utils.sandbox_state import SANDBOX_BACKENDS, get_sandbox_backend

logger = logging.getLogger(__name__)


async def setup_repo_node(state: dict, config: RunnableConfig) -> dict:
    """Clone repo, checkout branch, read AGENTS.md — deterministically.

    repo_cloned is ONLY set to True by this node. It is the precondition
    that downstream code-modifying nodes check.
    """
    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    repo = state.get("repo") or config.get("configurable", {}).get("repo")
    github_token = state.get("github_token")

    if not thread_id or not repo:
        return {
            "fatal_error": "Missing thread_id or repo in state",
            "error_stage": "setup_repo",
            "repo_cloned": False,
        }

    owner = repo.get("owner")
    name = repo.get("name")
    if not owner or not name:
        return {
            "fatal_error": "Missing repo.owner or repo.name",
            "error_stage": "setup_repo",
            "repo_cloned": False,
        }

    if not github_token:
        return {
            "fatal_error": "Missing github_token (entry node should have set it)",
            "error_stage": "setup_repo",
            "repo_cloned": False,
        }

    # Get sandbox backend
    try:
        sandbox = SANDBOX_BACKENDS.get(thread_id)
        if not sandbox:
            sandbox = await get_sandbox_backend(thread_id)
        if not sandbox:
            return {
                "fatal_error": f"No sandbox backend for thread {thread_id}",
                "error_stage": "setup_repo",
                "repo_cloned": False,
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "fatal_error": f"Failed to acquire sandbox: {exc}",
            "error_stage": "setup_repo",
            "repo_cloned": False,
        }

    # Resolve work dir
    try:
        work_dir = await aresolve_sandbox_work_dir(sandbox)
    except Exception as exc:  # noqa: BLE001
        return {
            "fatal_error": f"Failed to resolve work dir: {exc}",
            "error_stage": "setup_repo",
            "repo_cloned": False,
        }

    repo_dir = f"{work_dir}/{name}"
    safe_repo_dir = shlex.quote(repo_dir)

    # Determine branch name
    branch_name = (
        config.get("metadata", {}).get("branch_name")
        if isinstance(config.get("metadata"), dict)
        else None
    )
    if not branch_name:
        try:
            branch_name = await get_github_default_branch(owner, name, github_token)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch default branch: %s; falling back to 'main'", exc)
            branch_name = "main"

    # Check if repo already cloned
    check_result = await asyncio.to_thread(
        sandbox.execute, f"test -d {safe_repo_dir}/.git"
    )

    if check_result.exit_code == 0:
        # Already cloned — fetch latest
        logger.info("Repo already cloned at %s, fetching latest", repo_dir)
        fetch_result = await asyncio.to_thread(
            sandbox.execute,
            f"cd {safe_repo_dir} && git fetch origin 2>&1",
        )
        if fetch_result.exit_code != 0:
            logger.warning(
                "git fetch failed (continuing anyway): %s", fetch_result.output
            )
    else:
        # Fresh clone
        clone_url = (
            f"https://x-access-token:{github_token}@github.com/{owner}/{name}.git"
        )
        safe_clone_url = shlex.quote(clone_url)
        clone_cmd = f"git clone {safe_clone_url} {safe_repo_dir} 2>&1"
        logger.info("Cloning %s/%s into %s", owner, name, repo_dir)
        clone_result = await asyncio.to_thread(sandbox.execute, clone_cmd, timeout=300)
        if clone_result.exit_code != 0:
            return {
                "fatal_error": f"git clone failed: {clone_result.output}",
                "error_stage": "setup_repo",
                "repo_cloned": False,
            }

    # Checkout the branch (create if doesn't exist)
    safe_branch = shlex.quote(branch_name)
    # Try plain checkout first, then create
    checkout_result = await asyncio.to_thread(
        sandbox.execute,
        f"cd {safe_repo_dir} && git checkout {safe_branch} 2>&1",
    )
    if checkout_result.exit_code != 0:
        # Branch doesn't exist locally — try fetching it, then -B
        create_result = await asyncio.to_thread(
            sandbox.execute,
            f"cd {safe_repo_dir} && git checkout -B {safe_branch} 2>&1",
        )
        if create_result.exit_code != 0:
            return {
                "fatal_error": (
                    f"git checkout failed for branch {branch_name}: "
                    f"{create_result.output}"
                ),
                "error_stage": "setup_repo",
                "repo_cloned": False,
            }

    # Configure git identity (idempotent)
    await asyncio.to_thread(
        sandbox.execute,
        f"cd {safe_repo_dir} && "
        f"git config user.name 'native-swe[bot]' && "
        f"git config user.email 'native-swe@users.noreply.github.com'",
    )

    # Read AGENTS.md if present
    agents_md_content: str | None = None
    agents_md_path = f"{repo_dir}/AGENTS.md"
    safe_agents_path = shlex.quote(agents_md_path)
    agents_check = await asyncio.to_thread(
        sandbox.execute, f"test -f {safe_agents_path}"
    )
    if agents_check.exit_code == 0:
        cat_result = await asyncio.to_thread(
            sandbox.execute, f"cat {safe_agents_path}"
        )
        if cat_result.exit_code == 0:
            agents_md_content = cat_result.output
            logger.info("Read AGENTS.md (%d chars)", len(agents_md_content or ""))

    logger.info(
        "Setup complete: repo_dir=%s, branch=%s, AGENTS.md=%s",
        repo_dir,
        branch_name,
        "present" if agents_md_content else "absent",
    )

    return {
        "repo_cloned": True,
        "repo_dir": repo_dir,
        "branch_name": branch_name,
        "agents_md_content": agents_md_content,
        "files_changed_so_far": [],   # ADD — reset on each fresh setup
    }
