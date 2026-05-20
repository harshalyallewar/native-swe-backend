"""Main entry point for Native-SWE agent — uses the custom StateGraph."""

# ruff: noqa: E402
import logging
import warnings

from .logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="langchain_core._api.deprecation")
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

from langgraph.graph.state import RunnableConfig

from .graph.builder import build_graph

# Build graph once at module load.
_graph = build_graph()


async def get_agent(config: RunnableConfig):
    """LangGraph entrypoint — called per thread by the runtime.

    Returns the compiled graph configured with the per-thread runnable config.
    Sandbox creation, auth resolution, repo cloning, etc. all happen as graph
    nodes (entry_node and setup_repo_node).
    """
    return _graph.with_config(config)
