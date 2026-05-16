# tests/test_setup_flow.py
import pytest

from agent.nodes import entry_node, setup_repo_node


@pytest.mark.asyncio
async def test_entry_then_setup(monkeypatch):
    initial_state = {
        "messages": [],
        "source": "github",
        "repo": {"owner": "your-org", "name": "your-test-repo"},
        "thread_id": "test-thread-uuid",
        "repo_cloned": False,
        # ... rest of defaults
    }
    config = {"configurable": {"thread_id": "test-thread-uuid", "source": "github"}, "metadata": {}}

    entry_result = await entry_node(initial_state, config)
    assert "fatal_error" not in entry_result
    assert entry_result.get("github_token")
    assert entry_result.get("sandbox_id")

    merged = {**initial_state, **entry_result}
    setup_result = await setup_repo_node(merged, config)
    assert setup_result.get("repo_cloned") is True
    assert setup_result.get("repo_dir")
    assert setup_result.get("branch_name")
