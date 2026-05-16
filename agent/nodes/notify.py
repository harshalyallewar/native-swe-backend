"""Notify node: post final result to Slack/GitHub/Linear. Idempotent."""
from __future__ import annotations

import logging

from langchain_core.messages import AnyMessage
from langgraph.graph.state import RunnableConfig

from ..utils.github_app import get_github_app_installation_token
from ..utils.github_comments import post_github_comment
from ..utils.messages import extract_text_content
from ..utils.slack import post_slack_thread_reply

logger = logging.getLogger(__name__)


def _last_assistant_text(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai":
            text = extract_text_content(msg.content) or ""
            if text:
                return text
    return ""


def _build_summary(state: dict) -> str:
    plan = state.get("plan") or []
    if not plan:
        return ""
    lines = [f"- {t.get('description', '')}" for t in plan if t.get("description")]
    if not lines:
        return ""
    return "Summary of changes:\n" + "\n".join(lines)


async def notify_node(state: dict, config: RunnableConfig) -> dict:
    """Post the final outcome to the originating channel. Idempotent."""
    if state.get("notification_sent"):
        return {}

    source = state.get("source") or config.get("configurable", {}).get("source")
    fatal = state.get("fatal_error")
    pr_url = state.get("pr_url")
    verification_error = state.get("verification_error")
    intent = state.get("intent")
    messages = state.get("messages") or []

    # Compose message
    if fatal:
        message = (
            f"❌ Task failed at stage `{state.get('error_stage') or 'unknown'}`:\n{fatal}"
        )
    elif pr_url:
        summary = _build_summary(state)
        message = f"✅ Done! PR opened: {pr_url}"
        if summary:
            message = f"{message}\n\n{summary}"
    elif intent == "question":
        answer = _last_assistant_text(messages)
        message = answer or "Task completed."
    elif verification_error:
        attempts = state.get("verification_attempts", 0)
        message = (
            f"⚠️ Verification failed after {attempts} attempts:\n{verification_error}"
        )
    else:
        message = "Task completed."

    # Route to correct notification channel
    configurable = config.get("configurable", {}) if config else {}
    sent = False

    try:
        if source == "slack":
            slack_thread = configurable.get("slack_thread") or {}
            channel_id = slack_thread.get("channel_id")
            thread_ts = slack_thread.get("thread_ts")
            if channel_id and thread_ts:
                sent = await post_slack_thread_reply(channel_id, thread_ts, message)
            else:
                logger.warning("Slack source but missing channel_id/thread_ts")
        elif source == "github":
            repo = state.get("repo") or configurable.get("repo") or {}
            github_issue = configurable.get("github_issue") or {}
            issue_number = state.get("pr_number") or github_issue.get("number")
            if issue_number and repo.get("owner") and repo.get("name"):
                token = await get_github_app_installation_token()
                if token:
                    sent = await post_github_comment(
                        repo, int(issue_number), message, token=token
                    )
                else:
                    logger.error("No GitHub App token available for notify")
            else:
                logger.warning(
                    "GitHub source but missing issue_number or repo for notify"
                )
        elif source == "linear":
            # No Linear notify implemented in current codebase; log only
            logger.info("Linear source notification (no-op): %s", message[:100])
            sent = True
        else:
            logger.info("Unknown source '%s' for notify; message=%s", source, message[:100])
    except Exception:  # noqa: BLE001
        logger.exception("notify_node failed to send message")

    return {"notification_sent": True if sent else False}
