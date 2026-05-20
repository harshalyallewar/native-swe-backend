# LEGACY: This middleware is NOT wired into the current LangGraph state-machine
# architecture. Workspace validation is now handled by require_repo_cloned()
# in each graph node. Kept for reference only — do not import in new code.
import logging
import shlex
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from ..utils.sandbox_paths import aresolve_repo_dir
from ..utils.sandbox_state import get_sandbox_backend

logger = logging.getLogger(__name__)


class ExecutionStateMiddleware(AgentMiddleware):
    """Validates workspace state to ensure repository is cloned."""

    state_schema = AgentState

    async def _validate_workspace(self, tool_name: str, command: str) -> str | None:
        """Returns an error message if the workspace is invalid, None otherwise."""
        if tool_name not in ("execute", "commit_and_open_pr"):
            return None

        if tool_name == "execute" and ("git clone" in command or "list_repos" in command):
            return None

        config = get_config()
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        repo_config = configurable.get("repo", {})
        repo_name = repo_config.get("name")

        if not thread_id or not repo_name:
            return None

        sandbox_backend = await get_sandbox_backend(thread_id)
        if not sandbox_backend:
            return None

        repo_dir = await aresolve_repo_dir(sandbox_backend, repo_name)

        # Check if the repo directory and .git exist
        check_repo = await sandbox_backend.aexecute(f"test -d {shlex.quote(repo_dir)}/.git")
        if check_repo.exit_code != 0:
            return f"fatal: Repository not found at {repo_dir}. The directory does not exist or is not a git repository. You MUST use 'git clone' to clone the repository first before executing any other commands."

        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        # For synchronous execution, we skip workspace validation since it requires async
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_call = request.tool_call
        if isinstance(tool_call, dict):
            name = tool_call.get("name", "")
            args = tool_call.get("args", {})
            command = args.get("command", "") if name == "execute" else ""

            error_msg = await self._validate_workspace(name, command)
            if error_msg:
                # Return the error directly as a ToolMessage to prevent hallucination
                return ToolMessage(
                    content=error_msg,
                    name=name,
                    tool_call_id=tool_call.get("id", ""),
                )

        return await handler(request)
