"""Routing functions for the Native-SWE state graph."""

from __future__ import annotations

from .state import AgentState

MAX_VERIFICATION_ATTEMPTS = 3


def route_after_entry(state: AgentState) -> str:
    return "error" if state.get("fatal_error") else "continue"


def route_after_setup(state: AgentState) -> str:
    return "error" if state.get("fatal_error") else "continue"


def route_after_classify(state: AgentState) -> str:
    intent = state.get("intent")
    if intent in ("code_change", "question", "review"):
        return intent
    return "question"


def route_after_plan(state: AgentState) -> str:
    return "error" if state.get("fatal_error") else "continue"


def route_after_verify(state: AgentState) -> str:
    """
    After verify:
    - passed → commit_pr
    - failed but attempts remaining → re-run execute_step for fix
    - exhausted → notify
    """
    if state.get("lint_passed") and state.get("tests_passed"):
        return "passed"  # → commit_pr

    attempts = state.get("verification_attempts", 0)
    if attempts >= MAX_VERIFICATION_ATTEMPTS:
        return "exhausted"

    if state.get("intent") == "review":
        return "retry_review"

    return "retry"  # → execute_step to fix errors


def route_after_commit_pr(state: AgentState) -> str:
    # Always go to notify; notify reads pr_url and fatal_error
    # to compose the appropriate message.
    return "continue"
