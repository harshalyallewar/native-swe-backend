import asyncio
import logging
import os
from typing import Any

from exa_py import Exa
from langchain_core.runnables import RunnableConfig

from .parallel_search import search_parallel

logger = logging.getLogger(__name__)


async def web_search(
    query: str,
    num_results: int = 5,
    include_contents: bool = True,
    *,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Search the web for documentation, code examples, news, or research.

    Use this tool when you need to find documentation, code examples, GitHub repos,
    news, or research papers to help complete a task.

    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
        include_contents: Include page text (Parallel excerpts or Exa full text)

    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - results: Search results with source URLs, as text
        - error: Error message if something failed
    """
    api_key = os.environ.get("EXA_API_KEY")
    provider = os.environ.get("WEB_SEARCH_PROVIDER", "").strip().lower()
    if not provider:
        provider = "exa" if api_key else "parallel"
    if provider == "disabled":
        return {"success": False, "results": None, "error": "Web search is disabled."}
    if provider not in {"parallel", "exa"}:
        return {
            "success": False,
            "results": None,
            "error": "WEB_SEARCH_PROVIDER must be parallel, exa, or disabled.",
        }
    if not query.strip() or isinstance(num_results, bool) or num_results < 1:
        return {
            "success": False,
            "results": None,
            "error": "Provide a query and positive num_results.",
        }
    if provider == "exa" and not api_key:
        logger.warning("exa_api_key_missing")
        return {
            "success": False,
            "error": "EXA_API_KEY is not configured. Please add it to your environment variables.",
        }

    try:
        if provider == "parallel":
            thread_id = config.get("configurable", {}).get("thread_id")
            results = await search_parallel(query, num_results, include_contents, thread_id)
            return {"success": True, "results": results, "error": None}

        client = Exa(api_key=api_key)
        if include_contents:
            result = await asyncio.to_thread(
                client.search_and_contents,
                query,
                text=True,
                num_results=num_results,
                type="auto",
            )
        else:
            result = await asyncio.to_thread(
                client.search,
                query,
                num_results=num_results,
                type="auto",
            )
        return {"success": True, "results": str(result), "error": None}

    except Exception as e:
        logger.exception("web_search failed")
        return {"success": False, "results": None, "error": f"{type(e).__name__}: {e}"}
