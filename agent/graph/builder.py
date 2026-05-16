"""StateGraph builder for the Native-SWE custom state machine."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..nodes import (
    answer_qa_node,
    check_queue_node,
    classify_node,
    commit_pr_node,
    entry_node,
    execute_step_node,
    notify_node,
    plan_node,
    review_pr_node,
    setup_repo_node,
    verify_node,
)
from .routing import (
    route_after_classify,
    route_after_commit_pr,
    route_after_entry,
    route_after_plan,
    route_after_setup,
    route_after_verify,
)
from .state import AgentState


def build_graph():
    """Build and compile the Native-SWE state graph.

    Returns a compiled LangGraph StateGraph with a MemorySaver checkpointer.
    """
    g = StateGraph(AgentState)

    g.add_node("entry", entry_node)
    g.add_node("setup_repo", setup_repo_node)
    g.add_node("check_queue", check_queue_node)
    g.add_node("classify", classify_node)
    g.add_node("plan", plan_node)
    g.add_node("execute_step", execute_step_node)
    g.add_node("verify", verify_node)
    g.add_node("commit_pr", commit_pr_node)
    g.add_node("answer_qa", answer_qa_node)
    g.add_node("review_pr", review_pr_node)
    g.add_node("notify", notify_node)

    g.set_entry_point("entry")

    # entry → setup_repo | notify(error)
    g.add_conditional_edges(
        "entry",
        route_after_entry,
        {"continue": "setup_repo", "error": "notify"},
    )

    # setup_repo → check_queue | notify(error)
    g.add_conditional_edges(
        "setup_repo",
        route_after_setup,
        {"continue": "check_queue", "error": "notify"},
    )

    # check_queue → classify | verify (depends on caller context)
    # We route via has_new_instructions for execute_step return path,
    # but for initial entry from setup_repo we always go to classify.
    # The shared check_queue routing handles both: after setup, the
    # router still goes through "verify"... we need to distinguish.
    #
    # Solution: check_queue is called from multiple places. Use a
    # single conditional that defaults sensibly:
    #   - If we have no intent yet → classify (initial pass)
    #   - If has_new_instructions → classify (re-plan)
    #   - Otherwise → verify (post-execute_step path)
    g.add_conditional_edges(
        "check_queue",
        _route_check_queue_dispatch,
        {"classify": "classify", "verify": "verify", "notify": "notify"},
    )

    # classify → plan | answer_qa | review_pr
    g.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "code_change": "plan",
            "question": "answer_qa",
            "review": "review_pr",
        },
    )

    # plan → execute_step | notify(error)
    g.add_conditional_edges(
        "plan",
        route_after_plan,
        {"continue": "execute_step", "error": "notify"},
    )

    # execute_step → check_queue
    g.add_edge("execute_step", "check_queue")

    # verify → commit_pr | execute_step (retry) | review_pr (retry) | notify (exhausted)
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "passed": "commit_pr",
            "next_step": "execute_step",
            "retry": "execute_step",
            "retry_review": "review_pr",
            "exhausted": "notify",
        },
    )

    # commit_pr → notify
    g.add_conditional_edges(
        "commit_pr",
        route_after_commit_pr,
        {"continue": "notify", "error": "notify"},
    )

    # answer_qa → notify (read-only — skip verify/commit)
    g.add_edge("answer_qa", "notify")
    # review_pr → verify (changes were made, run lint/tests)
    g.add_edge("review_pr", "verify")
    # notify → END
    g.add_edge("notify", END)

    return g.compile(checkpointer=MemorySaver())


def _route_check_queue_dispatch(state: AgentState) -> str:
    """Dispatch from check_queue based on graph state.

    - No intent classified yet → go to classify (initial pass after setup_repo).
    - New instructions arrived mid-run → go to classify (re-classify).
    - Otherwise → go to verify (post-execute_step path).
    - Fatal error somehow set → notify.
    """
    if state.get("fatal_error"):
        return "notify"
    if state.get("has_new_instructions"):
        return "classify"
    if not state.get("intent"):
        return "classify"

    intent = state.get("intent")

     # question intent skips verify/commit — go straight to notify
    if intent == "question":
        return "notify"

    # code_change and review: go to verify
    return "verify"
