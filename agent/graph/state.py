from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RepoConfig(TypedDict):
    owner: str
    name: str


class TodoItem(TypedDict):
    description: str
    files_likely_touched: list[str]
    verification_strategy: str
    completed: bool
    files_actually_changed: list[str]  # populated by execute_step
    outcome_summary: str  # populated by execute_step


class AgentState(TypedDict):
    # ── inputs ──────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]
    source: Literal["github", "slack"]
    repo: RepoConfig
    thread_id: str

    # ── set by entry node ───────────────────────────────────
    github_token: str | None
    sandbox_id: str | None

    # ── set by setup_repo node (deterministic, never LLM) ───
    repo_cloned: bool
    repo_dir: str | None
    branch_name: str | None
    agents_md_content: str | None

    # ── set by classify node ─────────────────────────────────
    intent: Literal["code_change", "question", "review"] | None

    # ── set by plan node ─────────────────────────────────────
    plan: list[TodoItem]
    current_step: int

    # ── set by execute_step node ─────────────────────────────
    files_changed_so_far: list[str]

    # ── set by check_queue node ──────────────────────────────
    has_new_instructions: bool

    # ── set by verify node ───────────────────────────────────
    lint_passed: bool | None
    tests_passed: bool | None
    verification_attempts: int
    verification_error: str | None

    # ── set by commit_pr node ────────────────────────────────
    pr_url: str | None
    pr_number: int | None

    # ── set by notify node ───────────────────────────────────
    notification_sent: bool

    qa_answer: str | None  # explicit answer from answer_qa node
    pending_user_notification: str | None  # side message to send to user

    # ── error handling ───────────────────────────────────────
    fatal_error: str | None
    error_stage: str | None
