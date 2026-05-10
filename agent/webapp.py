"""Custom FastAPI routes for LangGraph server."""

import hashlib
import hmac
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from langchain_core.messages.content import create_text_block
from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient

from .utils.auth import (
    is_bot_token_only_mode,
    persist_encrypted_github_token,
    resolve_github_token_from_email,
)
from .utils.comments import get_recent_comments
from .utils.github_app import get_github_app_installation_token
from .utils.github_comments import (
    NATIVE_SWE_TAGS,
    build_pr_prompt,
    extract_pr_context,
    fetch_issue_comments,
    fetch_pr_comments_since_last_tag,
    format_github_comment_body_for_prompt,
    get_thread_id_from_branch,
    react_to_github_comment,
    sanitize_github_comment_body,
    verify_github_signature,
)
from .utils.github_token import get_github_token_from_thread
from .utils.github_user_email_map import GITHUB_USER_EMAIL_MAP
from .utils.multimodal import dedupe_urls, extract_image_urls, fetch_image_block
from .utils.repo import extract_repo_from_text
from .utils.sandbox import validate_sandbox_startup_config
from .utils.slack import (
    add_slack_reaction,
    fetch_slack_thread_messages,
    format_slack_messages_for_prompt,
    get_slack_user_info,
    get_slack_user_names,
    post_slack_trace_reply,
    resolve_slack_links_in_context,
    select_slack_context_messages,
    strip_bot_mention,
    verify_slack_signature,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    validate_sandbox_startup_config()
    yield


app = FastAPI(lifespan=lifespan)

GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
SLACK_BOT_USER_ID = os.environ.get("SLACK_BOT_USER_ID", "")
SLACK_BOT_USERNAME = os.environ.get("SLACK_BOT_USERNAME", "")
DEFAULT_REPO_OWNER = os.environ.get("DEFAULT_REPO_OWNER", "langchain-ai")
DEFAULT_REPO_NAME = os.environ.get("DEFAULT_REPO_NAME", "langchainplus")
SLACK_REPO_OWNER = os.environ.get("SLACK_REPO_OWNER", "") or DEFAULT_REPO_OWNER
SLACK_REPO_NAME = os.environ.get("SLACK_REPO_NAME", "") or DEFAULT_REPO_NAME

LANGGRAPH_URL = os.environ.get("LANGGRAPH_URL") or os.environ.get(
    "LANGGRAPH_URL_PROD", "http://localhost:2024"
)

_AGENT_VERSION_METADATA: dict[str, str] = (
    {"LANGSMITH_AGENT_VERSION": os.environ["LANGCHAIN_REVISION_ID"]}
    if os.environ.get("LANGCHAIN_REVISION_ID")
    else {}
)

ALLOWED_GITHUB_ORGS: frozenset[str] = frozenset(
    org.strip().lower()
    for org in os.environ.get("ALLOWED_GITHUB_ORGS", "").split(",")
    if org.strip()
)


_GITHUB_BOT_MESSAGE_PREFIXES = (
    "🔐 **GitHub Authentication Required**",
    "✅ **Pull Request Created**",
    "✅ **Pull Request Updated**",
    "**Pull Request Created**",
    "**Pull Request Updated**",
    "🤖 **Agent Response**",
    "❌ **Agent Error**",
)


def generate_thread_id_from_issue(issue_id: str) -> str:
    """Generate a deterministic thread ID from a Linear issue ID.

    Args:
        issue_id: The Linear issue ID

    Returns:
        A UUID-formatted thread ID derived from the issue ID
    """
    hash_bytes = hashlib.sha256(f"linear-issue:{issue_id}".encode()).hexdigest()
    return (
        f"{hash_bytes[:8]}-{hash_bytes[8:12]}-{hash_bytes[12:16]}-"
        f"{hash_bytes[16:20]}-{hash_bytes[20:32]}"
    )


def generate_thread_id_from_github_issue(issue_id: str) -> str:
    """Generate a deterministic thread ID from a GitHub issue ID."""
    hash_bytes = hashlib.sha256(f"github-issue:{issue_id}".encode()).hexdigest()
    return (
        f"{hash_bytes[:8]}-{hash_bytes[8:12]}-{hash_bytes[12:16]}-"
        f"{hash_bytes[16:20]}-{hash_bytes[20:32]}"
    )


def generate_thread_id_from_slack_thread(channel_id: str, thread_id: str) -> str:
    """Generate a deterministic thread ID from a Slack thread identifier."""
    composite = f"{channel_id}:{thread_id}"
    md5_hex = hashlib.md5(composite.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=md5_hex))


def _extract_repo_config_from_thread(thread: dict[str, Any]) -> dict[str, str] | None:
    """Extract repo config from persisted thread data."""
    metadata = thread.get("metadata")
    if not isinstance(metadata, dict):
        return None

    repo = metadata.get("repo")
    if isinstance(repo, dict):
        owner = repo.get("owner")
        name = repo.get("name")
        if isinstance(owner, str) and owner and isinstance(name, str) and name:
            return {"owner": owner, "name": name}

    owner = metadata.get("repo_owner")
    name = metadata.get("repo_name")
    if isinstance(owner, str) and owner and isinstance(name, str) and name:
        return {"owner": owner, "name": name}

    return None


def _is_not_found_error(exc: Exception) -> bool:
    """Best-effort check for LangGraph 404 errors."""
    return getattr(exc, "status_code", None) == 404


def _is_repo_org_allowed(repo_config: dict[str, str]) -> bool:
    """Check if the repo owner/org is in the allowlist.

    Returns True if no allowlist is configured (empty ALLOWED_GITHUB_ORGS),
    or if the repo owner is in the allowlist.
    """
    if not ALLOWED_GITHUB_ORGS:
        return True
    owner = repo_config.get("owner", "").lower()
    return owner in ALLOWED_GITHUB_ORGS


async def _upsert_slack_thread_repo_metadata(
    thread_id: str, repo_config: dict[str, str], langgraph_client: LangGraphClient
) -> None:
    """Persist the selected repo config on the thread metadata."""
    try:
        await langgraph_client.threads.update(thread_id=thread_id, metadata={"repo": repo_config})
    except Exception as exc:  # noqa: BLE001
        if _is_not_found_error(exc):
            try:
                await langgraph_client.threads.create(
                    thread_id=thread_id,
                    if_exists="do_nothing",
                    metadata={"repo": repo_config},
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to create Slack thread %s while persisting repo metadata",
                    thread_id,
                )
            return
        logger.exception(
            "Failed to persist Slack thread repo metadata for thread %s",
            thread_id,
        )


async def get_slack_repo_config(message: str, channel_id: str, thread_ts: str) -> dict[str, str]:
    """Resolve repository configuration for Slack-triggered runs."""
    default_owner = SLACK_REPO_OWNER.strip() or DEFAULT_REPO_OWNER
    default_name = SLACK_REPO_NAME.strip() or DEFAULT_REPO_NAME
    thread_id = generate_thread_id_from_slack_thread(channel_id, thread_ts)
    langgraph_client = get_client(url=LANGGRAPH_URL)

    repo_config = extract_repo_from_text(message, default_owner=default_owner)

    if not repo_config:
        try:
            thread = await langgraph_client.threads.get(thread_id)
            thread_repo_config = _extract_repo_config_from_thread(thread)
            if thread_repo_config:
                repo_config = thread_repo_config
        except Exception as exc:  # noqa: BLE001
            if not _is_not_found_error(exc):
                logger.exception(
                    "Failed to fetch Slack thread %s for repo resolution",
                    thread_id,
                )

    if not repo_config:
        repo_config = {"owner": default_owner, "name": default_name}

    return repo_config


async def is_thread_active(thread_id: str) -> bool:
    """Check if a thread is currently active (has a running run).

    Args:
        thread_id: The LangGraph thread ID

    Returns:
        True if the thread status is "busy", False otherwise
    """
    langgraph_client = get_client(url=LANGGRAPH_URL)
    try:
        logger.debug("Fetching thread status for %s from %s", thread_id, LANGGRAPH_URL)
        thread = await langgraph_client.threads.get(thread_id)
        status = thread.get("status", "idle")
        logger.info(
            "Thread %s status check: status=%s, is_busy=%s",
            thread_id,
            status,
            status == "busy",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Failed to get thread status for %s: %s (type: %s) - assuming not active",
            thread_id,
            e,
            type(e).__name__,
        )
        status = "idle"
    return status == "busy"


async def _thread_exists(thread_id: str) -> bool:
    """Return whether a LangGraph thread already exists."""
    langgraph_client = get_client(url=LANGGRAPH_URL)
    try:
        await langgraph_client.threads.get(thread_id)
        return True
    except Exception as exc:  # noqa: BLE001
        if _is_not_found_error(exc):
            return False
        logger.warning("Failed to fetch thread %s, assuming it exists", thread_id)
        return True


async def queue_message_for_thread(
    thread_id: str, message_content: str | list[dict[str, Any]] | dict[str, Any]
) -> bool:
    """Queue a message for a thread that is currently active.

    Stores the message in the langgraph store, namespaced to the thread.
    Supports multiple queued messages by storing them as a list (FIFO order).
    The before_model middleware will pick them up and inject them into state.

    Args:
        thread_id: The LangGraph thread ID
        message_content: The message content to queue (text or content blocks)

    Returns:
        True if successfully queued, False otherwise
    """
    langgraph_client = get_client(url=LANGGRAPH_URL)
    try:
        namespace = ("queue", thread_id)
        key = "pending_messages"

        new_message = {"content": message_content}

        existing_messages: list[dict[str, Any]] = []
        try:
            existing_item = await langgraph_client.store.get_item(namespace, key)
            if existing_item and existing_item.get("value"):
                existing_messages = existing_item["value"].get("messages", [])
        except Exception:  # noqa: BLE001
            logger.debug("No existing queued messages for thread %s", thread_id)

        existing_messages.append(new_message)
        value = {"messages": existing_messages}

        logger.info(
            "Attempting to queue message for thread %s (total queued: %d)",
            thread_id,
            len(existing_messages),
        )
        await langgraph_client.store.put_item(namespace, key, value)
        logger.info("Successfully queued message for thread %s", thread_id)
        return True  # noqa: TRY300
    except Exception:
        logger.exception("Failed to queue message for thread %s", thread_id)
        return False


async def process_slack_mention(event_data: dict[str, Any], repo_config: dict[str, str]) -> None:
    """Process a Slack app mention by creating or interrupting a thread run."""
    channel_id = event_data.get("channel_id", "")
    thread_ts = event_data.get("thread_ts", "")
    event_ts = event_data.get("event_ts", "")
    user_id = event_data.get("user_id", "")
    text = event_data.get("text", "")
    bot_user_id = event_data.get("bot_user_id", "")

    if not channel_id or not thread_ts or not event_ts:
        logger.warning(
            "Missing Slack event fields (channel_id=%s, thread_ts=%s, event_ts=%s)",
            channel_id,
            thread_ts,
            event_ts,
        )
        return

    reacted = await add_slack_reaction(channel_id, event_ts, "eyes")
    if not reacted:
        logger.debug(
            "Unable to add eyes reaction for Slack message ts=%s in channel=%s",
            event_ts,
            channel_id,
        )

    thread_id = generate_thread_id_from_slack_thread(channel_id, thread_ts)

    user_email = None
    user_name = ""
    if user_id:
        slack_user = await get_slack_user_info(user_id)
        if slack_user:
            profile = slack_user.get("profile", {})
            if isinstance(profile, dict):
                user_email = profile.get("email")
                user_name = (
                    profile.get("display_name")
                    or profile.get("real_name")
                    or slack_user.get("real_name")
                    or slack_user.get("name")
                    or ""
                )

    thread_messages = await fetch_slack_thread_messages(channel_id, thread_ts)
    if not any(str(message.get("ts")) == str(event_ts) for message in thread_messages):
        thread_messages.append({"ts": event_ts, "text": text, "user": user_id})

    context_messages, context_mode = select_slack_context_messages(
        thread_messages, event_ts, bot_user_id, SLACK_BOT_USERNAME
    )
    context_user_ids = [
        value
        for value in (message.get("user") for message in context_messages)
        if isinstance(value, str) and value
    ]
    user_names_by_id = await get_slack_user_names(context_user_ids)
    if user_id and user_name and user_id not in user_names_by_id:
        user_names_by_id[user_id] = user_name
    context_text = format_slack_messages_for_prompt(
        context_messages,
        user_names_by_id,
        bot_user_id=bot_user_id,
        bot_username=SLACK_BOT_USERNAME,
    )
    context_source = (
        "the previous message where I was tagged"
        if context_mode == "last_mention"
        else "the beginning of the thread"
    )
    clean_text = (
        strip_bot_mention(text, bot_user_id, bot_username=SLACK_BOT_USERNAME)
        or "(no text in mention)"
    )
    trigger_user = user_name or (f"<@{user_id}>" if user_id else "Unknown user")

    # Auto-resolve cross-posted Slack message links in context
    resolved_links_section, image_urls_from_links = await resolve_slack_links_in_context(
        context_messages, user_names_by_id
    )

    prompt = (
        "You were mentioned in Slack.\n\n"
        f"## Repository\n{repo_config.get('owner')}/{repo_config.get('name')}\n\n"
        f"## Triggered by\n{trigger_user}\n\n"
        f"## Slack Thread\n- Channel: {channel_id}\n- Thread TS: {thread_ts}\n"
        f"- Context starts at: {context_source}\n\n"
        f"## Conversation Context\n{context_text}\n\n"
        f"## Latest Mention Request\n{clean_text}\n\n"
        + (f"{resolved_links_section}\n\n" if resolved_links_section else "")
        + "Use `slack_thread_reply` to communicate in this Slack thread for clarifications, "
        "status updates, and final summaries. Use `slack_read_thread_messages` to read any "
        "Slack messages by providing channel_id and message_ts."
    )
    content_blocks: list[dict[str, Any]] = [create_text_block(prompt)]

    image_urls = dedupe_urls(
        [url for msg in context_messages for url in extract_image_urls(msg.get("text", ""))]
        + [
            f["url_private"]
            for msg in context_messages
            for f in msg.get("files", [])
            if isinstance(f, dict)
            and f.get("mimetype", "").startswith("image/")
            and f.get("url_private")
        ]
        + image_urls_from_links
    )
    if image_urls:
        logger.info("Preparing %d image(s) for Slack mention", len(image_urls))
        async with httpx.AsyncClient() as http_client:
            for image_url in image_urls:
                image_block = await fetch_image_block(image_url, http_client)
                if image_block:
                    content_blocks.append(image_block)

    configurable: dict[str, Any] = {
        "repo": repo_config,
        "slack_thread": {
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "triggering_user_id": user_id,
            "triggering_user_name": user_name,
            "triggering_user_email": user_email,
            "triggering_event_ts": event_ts,
        },
        "user_email": user_email,
        "source": "slack",
    }

    langgraph_client = get_client(url=LANGGRAPH_URL)
    await _upsert_slack_thread_repo_metadata(thread_id, repo_config, langgraph_client)

    thread_active = await is_thread_active(thread_id)
    if thread_active:
        logger.info(
            "Thread %s is active, queuing Slack message for middleware pickup",
            thread_id,
        )
        queued_payload = {"text": prompt, "image_urls": []}
        queued = await queue_message_for_thread(
            thread_id=thread_id,
            message_content=queued_payload,
        )
        if queued:
            logger.info("Slack message queued for thread %s", thread_id)
        else:
            logger.error("Failed to queue Slack message for thread %s", thread_id)
        return

    await langgraph_client.runs.create(
        thread_id,
        "agent",
        input={"messages": [{"role": "user", "content": content_blocks}]},
        config={"configurable": configurable, "metadata": _AGENT_VERSION_METADATA},
        if_not_exists="create",
        multitask_strategy="interrupt",
    )
    await post_slack_trace_reply(channel_id, thread_ts, thread_id)


@app.post("/webhooks/slack")
async def slack_webhook(request: Request, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Handle Slack Event API webhooks for app mentions."""
    body = await request.body()

    signature = request.headers.get("X-Slack-Signature", "")
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    if not verify_slack_signature(
        body=body,
        timestamp=timestamp,
        signature=signature,
        secret=SLACK_SIGNING_SECRET,
    ):
        logger.warning("Invalid Slack signature")
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        logger.exception("Failed to parse Slack webhook JSON")
        return {"status": "error", "message": "Invalid JSON"}

    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge", "")
        return {"challenge": challenge}

    if payload.get("type") != "event_callback":
        return {"status": "ignored", "reason": "Not an event callback"}

    event = payload.get("event", {})
    if event.get("type") != "app_mention":
        message_text = event.get("text", "")
        has_username_mention = bool(
            event.get("type") == "message"
            and SLACK_BOT_USERNAME
            and f"@{SLACK_BOT_USERNAME}" in message_text
        )
        has_id_mention = bool(
            event.get("type") == "message"
            and SLACK_BOT_USER_ID
            and f"<@{SLACK_BOT_USER_ID}>" in message_text
        )
        if not (has_username_mention or has_id_mention):
            return {"status": "ignored", "reason": "Not an app_mention event"}

    if event.get("subtype") == "bot_message" or event.get("bot_id"):
        return {"status": "ignored", "reason": "Event from a bot"}

    channel_id = event.get("channel", "")
    event_ts = event.get("ts", "")
    thread_ts = event.get("thread_ts") or event_ts
    user_id = event.get("user", "")
    text = event.get("text", "")
    if not channel_id or not event_ts or not thread_ts:
        return {"status": "ignored", "reason": "Missing channel/thread timestamp"}

    bot_user_id = SLACK_BOT_USER_ID
    if not bot_user_id:
        authorizations = payload.get("authorizations", [])
        if isinstance(authorizations, list) and authorizations:
            auth_user_id = authorizations[0].get("user_id")
            if isinstance(auth_user_id, str):
                bot_user_id = auth_user_id
    if not bot_user_id:
        authed_users = payload.get("authed_users", [])
        if isinstance(authed_users, list) and authed_users:
            first_user = authed_users[0]
            if isinstance(first_user, str):
                bot_user_id = first_user

    if bot_user_id and user_id == bot_user_id:
        return {"status": "ignored", "reason": "Event from this bot user"}

    event_data = {
        "channel_id": channel_id,
        "thread_ts": thread_ts,
        "event_ts": event_ts,
        "user_id": user_id,
        "text": text,
        "bot_user_id": bot_user_id,
    }
    repo_config = await get_slack_repo_config(text, channel_id, thread_ts)

    if not _is_repo_org_allowed(repo_config):
        logger.warning(
            "Rejecting Slack webhook: org '%s' not in ALLOWED_GITHUB_ORGS",
            repo_config.get("owner"),
        )
        return {"status": "ignored", "reason": "Repository org not in allowlist"}

    background_tasks.add_task(process_slack_mention, event_data, repo_config)

    return {"status": "accepted", "message": "Slack mention queued"}


@app.get("/webhooks/slack")
async def slack_webhook_verify() -> dict[str, str]:
    """Verify endpoint for Slack webhook setup."""
    return {"status": "ok", "message": "Slack webhook endpoint is active"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


_SUPPORTED_GH_EVENTS = frozenset(
    ["issue_comment", "issues", "pull_request_review_comment", "pull_request_review"]
)
_SUPPORTED_GH_ISSUE_ACTIONS = frozenset(["edited", "opened", "reopened"])


def _build_github_issue_comments_text(comments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for comment in comments:
        body = comment.get("body", "")
        if not body or any(body.startswith(prefix) for prefix in _GITHUB_BOT_MESSAGE_PREFIXES):
            continue
        author = comment.get("author", "unknown")
        formatted_body = format_github_comment_body_for_prompt(author, body)
        lines.append(f"\n**{author}:**\n{formatted_body}\n")

    if not lines:
        return ""
    return "\n\n## Comments:\n" + "".join(lines)


def build_github_issue_prompt(
    repo_config: dict[str, str],
    issue_number: int,
    issue_id: str,
    title: str,
    body: str,
    comments: list[dict[str, Any]],
    *,
    github_login: str,
    issue_author: str = "",
) -> str:
    """Build the user prompt for a GitHub issue-triggered run."""
    triggered_by_line = f"## Triggered by: {github_login}\n\n" if github_login else ""
    comments_text = _build_github_issue_comments_text(comments)
    sanitized_title = sanitize_github_comment_body(title)
    formatted_body = format_github_comment_body_for_prompt(issue_author or github_login, body)
    return (
        "Please work on the following GitHub issue:\n\n"
        f"## Repository: {repo_config.get('owner')}/{repo_config.get('name')}\n\n"
        f"{triggered_by_line}"
        f"## GitHub Issue: #{issue_number} - Issue ID: {issue_id}\n\n"
        f"## Title: {sanitized_title}\n\n"
        f"## Description:\n{formatted_body}\n"
        f"{comments_text}\n\n"
        "Please analyze this issue and implement the necessary changes. "
        "When you need to communicate on GitHub, use `github_comment` with the issue number."
    )


def build_github_issue_followup_prompt(github_login: str, comment_body: str) -> str:
    """Build the prompt for a follow-up GitHub issue comment."""
    return (
        f"**{github_login}:**\n{format_github_comment_body_for_prompt(github_login, comment_body)}"
    )


def build_github_issue_update_prompt(github_login: str, title: str, body: str) -> str:
    """Build the prompt for a follow-up GitHub issue title/body update."""
    sanitized_title = sanitize_github_comment_body(title)
    formatted_body = format_github_comment_body_for_prompt(github_login, body)
    return (
        f"**{github_login}:** updated the GitHub issue title/body.\n\n"
        f"Title: {sanitized_title}\n\n"
        f"Description:\n{formatted_body}"
    )


async def _trigger_or_queue_run(
    thread_id: str,
    prompt: str,
    *,
    github_login: str,
    github_user_id: int | None,
    repo_config: dict[str, str],
    pr_number: int,
) -> None:
    """Create a new agent run or queue the message if the thread is busy."""
    thread_active = await is_thread_active(thread_id)
    if thread_active:
        logger.info("Thread %s is busy, queuing GitHub PR comment message", thread_id)
        await queue_message_for_thread(thread_id, prompt)
        return

    logger.info("Creating LangGraph run for thread %s from GitHub PR comment", thread_id)
    langgraph_client = get_client(url=LANGGRAPH_URL)
    await langgraph_client.runs.create(
        thread_id,
        "agent",
        input={"messages": [{"role": "user", "content": prompt}]},
        config={
            "configurable": {
                "source": "github",
                "github_login": github_login,
                "github_user_id": github_user_id,
                "repo": repo_config,
                "pr_number": pr_number,
            },
            "metadata": _AGENT_VERSION_METADATA,
        },
        if_not_exists="create",
    )
    logger.info("LangGraph run created for thread %s from GitHub PR comment", thread_id)


async def _get_or_resolve_thread_github_token(thread_id: str, email: str) -> str | None:
    """Resolve and persist a GitHub token for a thread when available.

    In bot-token-only mode, returns a fresh GitHub App installation token
    instead of resolving per-user OAuth tokens.
    """
    if is_bot_token_only_mode():
        bot_token = await get_github_app_installation_token()
        if bot_token:
            try:
                await persist_encrypted_github_token(thread_id, bot_token)
            except Exception:
                logger.warning("Could not persist bot token for thread %s", thread_id)
            return bot_token
        logger.warning("Bot-token-only mode but GitHub App token unavailable")
        return None

    github_token, _encrypted_token = await get_github_token_from_thread(thread_id)
    if github_token:
        return github_token

    auth_result = await resolve_github_token_from_email(email)
    github_token = auth_result.get("token")
    if not github_token:
        return None

    try:
        await persist_encrypted_github_token(thread_id, github_token)
    except Exception:
        logger.warning("Could not persist GitHub token for thread %s", thread_id)
    return github_token


async def process_github_pr_comment(payload: dict[str, Any], event_type: str) -> None:
    """Process a GitHub PR comment that tagged @native-swe.

    Retrieves the existing thread token, reacts with 👀, fetches all comments
    since the last @native-swe tag, then creates or queues a new run.

    Args:
        payload: The parsed GitHub webhook payload.
        event_type: One of 'issue_comment', 'pull_request_review_comment',
                    'pull_request_review'.
    """
    (
        repo_config,
        pr_number,
        branch_name,
        github_login,
        pr_url,
        comment_id,
        node_id,
    ) = await extract_pr_context(payload, event_type)
    github_user_id = payload.get("sender", {}).get("id")

    logger.info(
        "Processing GitHub PR comment: event=%s, pr=%s, branch=%s",
        event_type,
        pr_number,
        branch_name,
    )

    thread_id = get_thread_id_from_branch(branch_name) if branch_name else None
    if not thread_id:
        if not pr_number:
            logger.warning(
                "Could not determine thread_id for branch '%s' (no pr_number), skipping",
                branch_name,
            )
            return
        owner = repo_config.get("owner", "")
        name = repo_config.get("name", "")
        stable_key = f"{owner}/{name}/pr/{pr_number}"
        thread_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        logger.info("Generated thread_id %s for non-native-swe branch '%s'", thread_id, branch_name)
        langgraph_client = get_client(url=LANGGRAPH_URL)
        try:
            await langgraph_client.threads.update(thread_id, metadata={"branch_name": branch_name})
        except Exception as exc:  # noqa: BLE001
            if _is_not_found_error(exc):
                await langgraph_client.threads.create(
                    thread_id=thread_id,
                    if_exists="do_nothing",
                    metadata={"branch_name": branch_name},
                )
            else:
                logger.warning("Failed to persist branch_name metadata for thread %s", thread_id)

    email = GITHUB_USER_EMAIL_MAP.get(github_login, "")
    if not email:
        logger.warning("No email mapping for GitHub user '%s', skipping", github_login)
        return

    github_token = await _get_or_resolve_thread_github_token(thread_id, email)
    if not github_token:
        logger.warning("No GitHub token for thread %s, skipping", thread_id)
        return

    if comment_id:
        await react_to_github_comment(
            repo_config,
            comment_id,
            event_type=event_type,
            token=github_token,
            pull_number=pr_number,
            node_id=node_id,
        )

    if not pr_number:
        logger.warning("No PR number found in payload, skipping")
        return

    comments = await fetch_pr_comments_since_last_tag(repo_config, pr_number, token=github_token)
    if not comments:
        logger.info("No comments found since last @native-swe tag for PR %s", pr_number)
        return

    prompt = build_pr_prompt(comments, pr_url, repo_config=repo_config)
    await _trigger_or_queue_run(
        thread_id,
        prompt,
        github_login=github_login,
        github_user_id=github_user_id,
        repo_config=repo_config,
        pr_number=pr_number,
    )


async def process_github_issue(payload: dict[str, Any], event_type: str) -> None:
    """Process a GitHub issue or issue comment that tagged @native-swe."""
    issue = payload.get("issue", {})
    repo = payload.get("repository", {})
    repo_config = {
        "owner": repo.get("owner", {}).get("login", ""),
        "name": repo.get("name", ""),
    }


    issue_id = str(issue.get("id", ""))
    issue_number = issue.get("number")
    github_login = payload.get("sender", {}).get("login", "")
    github_user_id = payload.get("sender", {}).get("id")
    issue_url = issue.get("html_url", "") or issue.get("url", "")
    title = issue.get("title", "No title")
    description = issue.get("body") or "No description"
    issue_author = issue.get("user", {}).get("login", "")

    logger.info(
        "Processing GitHub issue: event=%s, issue=%s, repo=%s/%s",
        event_type,
        issue_number,
        repo_config.get("owner"),
        repo_config.get("name"),
    )

    if not issue_id or not issue_number:
        logger.warning("Missing GitHub issue id/number, skipping")
        return

    email = GITHUB_USER_EMAIL_MAP.get(github_login, "")
    if not email:
        logger.warning("No email mapping for GitHub user '%s', skipping", github_login)
        return

    thread_id = generate_thread_id_from_github_issue(issue_id)
    existing_thread = await _thread_exists(thread_id)
    github_token = await _get_or_resolve_thread_github_token(thread_id, email)
    app_token = await get_github_app_installation_token()
    reaction_token = github_token or app_token
    comment = payload.get("comment", {})
    comment_id = comment.get("id")
    if event_type == "issue_comment" and comment_id:
        if not reaction_token:
            logger.warning("No GitHub token available to react to issue comment %s", comment_id)
        else:
            reacted = await react_to_github_comment(
                repo_config,
                comment_id,
                event_type="issue_comment",
                token=reaction_token,
            )
            if not reacted:
                logger.warning("Failed to react to GitHub issue comment %s", comment_id)

    if existing_thread:
        if event_type == "issue_comment":
            prompt = build_github_issue_followup_prompt(
                comment.get("user", {}).get("login", github_login) or github_login,
                comment.get("body", ""),
            )
        else:
            prompt = build_github_issue_update_prompt(github_login, title, description)
    else:
        comments = await fetch_issue_comments(
            repo_config, issue_number, token=github_token or app_token
        )
        if comment_id and not any(item.get("comment_id") == comment_id for item in comments):
            comments.append(
                {
                    "body": comment.get("body", ""),
                    "author": comment.get("user", {}).get("login", "unknown"),
                    "created_at": comment.get("created_at", ""),
                    "comment_id": comment_id,
                }
            )
            comments.sort(key=lambda item: item.get("created_at", ""))

        prompt = build_github_issue_prompt(
            repo_config,
            issue_number,
            issue_id,
            title,
            description,
            comments,
            github_login=github_login,
            issue_author=issue_author,
        )
    configurable: dict[str, Any] = {
        "source": "github",
        "github_login": github_login,
        "github_user_id": github_user_id,
        "repo": repo_config,
        "github_issue": {
            "id": issue_id,
            "number": issue_number,
            "title": title,
            "url": issue_url,
        },
    }

    thread_active = await is_thread_active(thread_id)
    if thread_active:
        logger.info("Thread %s is busy, queuing GitHub issue message", thread_id)
        await queue_message_for_thread(thread_id, prompt)
        return

    logger.info("Creating LangGraph run for thread %s from GitHub issue", thread_id)
    langgraph_client = get_client(url=LANGGRAPH_URL)
    await langgraph_client.runs.create(
        thread_id,
        "agent",
        input={"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": configurable, "metadata": _AGENT_VERSION_METADATA},
        if_not_exists="create",
    )
    logger.info("LangGraph run created for thread %s from GitHub issue", thread_id)


@app.post("/webhooks/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Handle GitHub webhooks for issue and PR events that tag @native-swe."""
    body = await request.body()


    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_github_signature(body, signature, secret=GITHUB_WEBHOOK_SECRET):
        logger.warning("Invalid GitHub webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")

    event_type = request.headers.get("X-GitHub-Event", "")
    if event_type not in _SUPPORTED_GH_EVENTS:
        logger.info("Ignoring unsupported GitHub event type: %s", event_type)
        return {"status": "ignored", "reason": f"Unsupported event type: {event_type}"}

    try:
        payload = json.loads(body)

    except json.JSONDecodeError:
        logger.exception("Failed to parse GitHub webhook JSON")
        return {"status": "error", "message": "Invalid JSON"}

    # Check org allowlist
    webhook_repo = payload.get("repository", {})
    webhook_repo_config = {
        "owner": webhook_repo.get("owner", {}).get("login", ""),
        "name": webhook_repo.get("name", ""),
    }
    if not _is_repo_org_allowed(webhook_repo_config):
        logger.warning(
            "Rejecting GitHub webhook: org '%s' not in ALLOWED_GITHUB_ORGS",
            webhook_repo_config.get("owner"),
        )
        return {"status": "ignored", "reason": "Repository org not in allowlist"}

    issue = payload.get("issue", {})
    is_pull_request_comment = bool(event_type == "issue_comment" and issue.get("pull_request"))
    is_issue_comment = bool(event_type == "issue_comment" and not issue.get("pull_request"))
    is_issue_event = event_type == "issues"

    if is_issue_event:
        action = payload.get("action", "")
        if action not in _SUPPORTED_GH_ISSUE_ACTIONS:
            logger.info("Ignoring unsupported GitHub issue action: %s", action)
            return {"status": "ignored", "reason": f"Unsupported GitHub issue action: {action}"}
        if action == "edited":
            
            changes = payload.get("changes", {})
            if not any(field in changes for field in ("body", "title")):
                logger.info("Ignoring GitHub issue edit without title/body changes")
                return {"status": "ignored", "reason": "Issue edit did not change title or body"}

        issue_text = f"{issue.get('title', '')}\n\n{issue.get('body', '')}".lower()
        if not any(tag in issue_text for tag in NATIVE_SWE_TAGS):
            logger.info("Ignoring issue event that does not mention @native-swe")
            return {"status": "ignored", "reason": "Issue does not mention @native-swe"}

        logger.info("Accepted GitHub issue webhook, scheduling background task")
        background_tasks.add_task(process_github_issue, payload, event_type)
        return {"status": "accepted", "message": "Processing GitHub issue event"}

    comment = payload.get("comment") or payload.get("review", {})
    comment_body = (comment.get("body") or "") if comment else ""
    if not any(tag in comment_body.lower() for tag in NATIVE_SWE_TAGS):
        logger.info("Ignoring comment that does not mention @native-swe")
        return {"status": "ignored", "reason": "Comment does not mention @native-swe"}

    logger.info("Accepted GitHub webhook: event=%s, scheduling background task", event_type)
    if is_pull_request_comment or event_type in {
        "pull_request_review_comment",
        "pull_request_review",
    }:
        background_tasks.add_task(process_github_pr_comment, payload, event_type)
        return {"status": "accepted", "message": f"Processing {event_type} event"}

    if is_issue_comment:
        background_tasks.add_task(process_github_issue, payload, event_type)
        return {"status": "accepted", "message": "Processing GitHub issue comment event"}

    logger.info("Ignoring unsupported GitHub payload shape for event=%s", event_type)
    return {"status": "ignored", "reason": f"Unsupported payload for event type: {event_type}"}
