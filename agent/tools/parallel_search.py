"""Anonymous Parallel Search MCP through the official Streamable HTTP client."""

import hashlib
import json
from datetime import timedelta

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "Native-SWE/0.1.0 (+https://github.com/harshalyallewar/native-swe-backend)"
SEARCH_TIMEOUT = 30


async def search_parallel(
    query: str, num_results: int, include_contents: bool, thread_id: str | None
) -> str:
    arguments = {"objective": query, "search_queries": [query]}
    if thread_id:
        # Keep a stable conversation identity without sending internal thread names.
        arguments["session_id"] = hashlib.sha256(thread_id.encode()).hexdigest()

    with anyio.fail_after(SEARCH_TIMEOUT):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=SEARCH_TIMEOUT
        ) as client:
            async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                async with ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=SEARCH_TIMEOUT)
                ) as session:
                    await session.initialize()
                    result = await session.call_tool("web_search", arguments)

    if result.isError:
        message = "\n".join(block.text for block in result.content if block.type == "text")
        raise ValueError(message or "Parallel Search MCP returned an error")
    payload = result.structuredContent
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Parallel Search MCP returned an invalid search response")
    rows = payload["results"][:num_results]
    if not include_contents:
        rows = [{k: v for k, v in row.items() if k != "excerpts"} for row in rows]
    return json.dumps(rows, ensure_ascii=False)
