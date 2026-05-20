"""Commit and open PR node: deterministic, no LLM."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage
from langgraph.graph.state import RunnableConfig

from ..utils.authorship import (
    NATIVE_SWE_BOT_EMAIL,
    NATIVE_SWE_BOT_NAME,
    add_pr_collaboration_note,
    add_user_coauthor_trailer,
    resolve_triggering_user_identity,
)
from ..utils.github import (
    create_github_pr,
    get_github_default_branch,
    git_add_all,
    git_checkout_branch,
    git_commit,
    git_config_user,
    git_current_branch,
    git_fetch_origin,
    git_has_uncommitted_changes,
    git_has_unpushed_commits,
    git_push,
    is_permanent_github_push_failure,
)
from ..utils.github_app import get_github_app_installation_token
from ..utils.sandbox_state import get_sandbox_backend

logger = logging.getLogger(__name__)


def _build_commit_message(state: dict) -> str:
    """Build commit message from plan summary or fallback."""
    plan = state.get("plan") or []
    if plan:
        descriptions = [t.get("description", "") for t in plan if t.get("description")]
        if descriptions:
            return (
                descriptions[0]
                if len(descriptions) == 1
                else (f"{descriptions[0]}\n\n" + "\n".join(f"- {d}" for d in descriptions[1:]))
            )
    return "Automated changes by native-swe"


def _build_pr_title(state: dict) -> str:
    plan = state.get("plan") or []
    if plan and plan[0].get("description"):
        title = plan[0]["description"].strip().split("\n")[0]
        if len(title) > 70:
            title = title[:67] + "..."
        # Lowercase first letter to fit convention
        return (
            f"feat: {title.lower()}"
            if not title.lower().startswith(("fix:", "feat:", "chore:", "ci:"))
            else title
        )
    return "feat: native-swe automated changes"


def _build_pr_body(state: dict) -> str:
    plan = state.get("plan") or []
    if not plan:
        return (
            "## Description\nAutomated changes by native-swe.\n\n## Test Plan\n- [ ] Review changes"
        )
    descriptions = "\n".join(f"- {t.get('description', '')}" for t in plan if t.get("description"))
    return (
        "## Description\n"
        f"{descriptions}\n\n"
        "## Release Note\nnone\n\n"
        "## Test Plan\n- [ ] Review automated changes"
    )


async def commit_pr_node(state: dict, config: RunnableConfig) -> dict:
    """Commit pending changes, push, and open/update a draft PR."""
    thread_id = state.get("thread_id") or config.get("configurable", {}).get("thread_id")
    repo = state.get("repo") or config.get("configurable", {}).get("repo") or {}
    repo_dir = state.get("repo_dir")
    branch_name = state.get("branch_name")
    github_token = state.get("github_token")

    if not thread_id or not repo or not repo_dir:
        return {
            "fatal_error": "Missing thread_id/repo/repo_dir for commit_pr",
            "error_stage": "commit_pr",
        }

    repo_owner = repo.get("owner")
    repo_name = repo.get("name")
    if not repo_owner or not repo_name:
        return {
            "fatal_error": "Missing repo owner/name for commit_pr",
            "error_stage": "commit_pr",
        }

    sandbox = await get_sandbox_backend(thread_id)
    if not sandbox:
        return {
            "fatal_error": "No sandbox available for commit_pr",
            "error_stage": "commit_pr",
        }

    # Check for changes
    has_uncommitted = await asyncio.to_thread(git_has_uncommitted_changes, sandbox, repo_dir)
    await asyncio.to_thread(git_fetch_origin, sandbox, repo_dir)
    has_unpushed = await asyncio.to_thread(git_has_unpushed_commits, sandbox, repo_dir)

    if not (has_uncommitted or has_unpushed):
        logger.info("No changes detected — skipping commit/PR")
        return {
            "pr_url": None,
            "pr_number": None,
            "messages": [
                HumanMessage(
                    content="[commit_pr] No changes detected — task may have been read-only."
                )
            ],
        }

    # Identity for git + co-author trailer
    user_identity = await asyncio.to_thread(resolve_triggering_user_identity, config, github_token)

    # Configure git user
    await asyncio.to_thread(
        git_config_user, sandbox, repo_dir, NATIVE_SWE_BOT_NAME, NATIVE_SWE_BOT_EMAIL
    )

    # Build commit message + PR metadata
    raw_commit_message = _build_commit_message(state)
    commit_message = add_user_coauthor_trailer(raw_commit_message, user_identity)
    pr_title = _build_pr_title(state)
    pr_body = add_pr_collaboration_note(_build_pr_body(state), user_identity)

    # Resolve target branch
    target_branch = branch_name or f"native-swe/{thread_id}"
    current_branch = await asyncio.to_thread(git_current_branch, sandbox, repo_dir)
    if current_branch != target_branch:
        ok, err = await asyncio.to_thread(git_checkout_branch, sandbox, repo_dir, target_branch)
        if not ok:
            return {
                "fatal_error": f"git checkout {target_branch} failed: {err}",
                "error_stage": "commit_pr",
            }

    # Stage + commit
    if has_uncommitted:
        await asyncio.to_thread(git_add_all, sandbox, repo_dir)
        commit_result = await asyncio.to_thread(git_commit, sandbox, repo_dir, commit_message)
        if commit_result.exit_code != 0:
            logger.error("git commit failed: %s", commit_result.output)
            return {
                "messages": [
                    HumanMessage(
                        content=f"[commit_pr] git commit failed: {commit_result.output.strip()}"
                    )
                ],
                "pr_url": None,
                "pr_number": None,
            }

    # Get installation token for push + PR
    installation_token = await get_github_app_installation_token()
    if not installation_token:
        return {
            "fatal_error": "Failed to get GitHub App installation token",
            "error_stage": "commit_pr",
        }

    # Push
    push_result = await asyncio.to_thread(git_push, sandbox, repo_dir, target_branch)
    if push_result.exit_code != 0:
        push_output = push_result.output.strip()
        if is_permanent_github_push_failure(push_output):
            return {
                "fatal_error": (
                    f"permanent push failure: token does not have write access. {push_output}"
                ),
                "error_stage": "commit_pr",
            }
        return {
            "messages": [HumanMessage(content=f"[commit_pr] git push failed: {push_output}")],
            "pr_url": None,
            "pr_number": None,
        }

    # Resolve base + open PR
    try:
        base_branch = await get_github_default_branch(repo_owner, repo_name, installation_token)
    except Exception:  # noqa: BLE001
        base_branch = "main"

    try:
        pr_url, pr_number, _existing = await create_github_pr(
            repo_owner=repo_owner,
            repo_name=repo_name,
            github_token=github_token or installation_token,
            title=pr_title,
            head_branch=target_branch,
            base_branch=base_branch,
            body=pr_body,
            installation_token=installation_token,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to create GitHub PR")
        return {
            "messages": [HumanMessage(content=f"[commit_pr] PR creation failed: {exc}")],
            "pr_url": None,
            "pr_number": None,
        }

    if not pr_url:
        return {
            "messages": [HumanMessage(content="[commit_pr] PR creation returned no URL")],
            "pr_url": None,
            "pr_number": None,
        }

    logger.info("Opened PR: %s (#%s)", pr_url, pr_number)
    return {
        "pr_url": pr_url,
        "pr_number": pr_number,
        "files_changed_so_far": [],  # ADD — clean slate after commit
    }
