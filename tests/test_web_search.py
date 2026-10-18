"""Exercise the normal agent tool dispatcher and the real MCP HTTP client.

The chat model and remote server are scripted; no model inference or paid keys
are used. HTTP requests still pass through MCP initialization and tools/call.
"""

import asyncio
import hashlib
import importlib
import json

import httpx
import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

parallel = importlib.import_module("agent.tools.parallel_search")
search = importlib.import_module("agent.tools.web_search")
qa = importlib.import_module("agent.nodes.answer_qa")

CONFIG = {"configurable": {"thread_id": "search-test-thread"}}
ROWS = [
    {"url": "https://docs.python.org/3/", "title": "Python docs", "excerpts": ["Python reference"]},
    {"url": "https://example.org/docs", "title": "More docs", "excerpts": ["More context"]},
]


@pytest.fixture(autouse=True)
def isolated_search_environment(monkeypatch):
    for name in ("EXA_API_KEY", "PARALLEL_API_KEY", "WEB_SEARCH_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def mcp_server(monkeypatch):
    requests = []
    state = {"result": {"results": ROWS}, "error": False, "delay": False}

    async def respond(request):
        requests.append(request)
        assert str(request.url) == parallel.MCP_URL
        assert request.headers["user-agent"] == parallel.USER_AGENT
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        method = payload["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "scripted-search", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            if state["delay"]:
                await asyncio.Event().wait()
            if state.get("http_error"):
                return httpx.Response(state["http_error"])
            result = {
                "content": [{"type": "text", "text": json.dumps(state["result"])}],
                "structuredContent": state["result"],
                "isError": state["error"],
            }
        else:
            raise AssertionError(f"Unexpected MCP method: {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        parallel.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    return requests, state


def dispatcher(tools):
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


async def dispatch(args=None):
    node = dispatcher([tool(search.web_search)])
    result = await node.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": args or {"query": "Python reference documentation"},
                            "id": "search-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=CONFIG,
    )
    return json.loads(result["messages"][-1].content)


async def test_anonymous_dispatch_preserves_urls_and_controls(mcp_server):
    requests, _ = mcp_server
    result = await dispatch(
        {"query": "Python reference documentation", "num_results": 1, "include_contents": False}
    )
    assert result["success"] is True
    assert json.loads(result["results"]) == [{"url": ROWS[0]["url"], "title": "Python docs"}]
    payloads = [json.loads(r.content) for r in requests]
    call = next(p for p in payloads if p["method"] == "tools/call")
    assert call["params"]["name"] == "web_search"
    assert call["params"]["arguments"] == {
        "objective": "Python reference documentation",
        "search_queries": ["Python reference documentation"],
        "session_id": hashlib.sha256(b"search-test-thread").hexdigest(),
    }
    assert "initialize" in [p["method"] for p in payloads]
    assert "tools/list" in [p["method"] for p in payloads]


class ScriptedSearchModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "scripted-search-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage) and m.name == "web_search"]
        if results:
            payload = json.loads(results[-1].content)
            assert payload["success"] is True
            row = json.loads(payload["results"])[0]
            message = AIMessage(content=f"{row['excerpts'][0]}: {row['url']}")
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "Python reference documentation"},
                        "id": "search-call",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_qa_node_consumes_search_result_in_real_agent_loop(monkeypatch, mcp_server, tmp_path):
    async def backend(thread_id):
        return FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    monkeypatch.setattr(qa, "get_sandbox_backend", backend)
    monkeypatch.setattr(qa, "make_model", lambda *args, **kwargs: ScriptedSearchModel())
    result = await qa.answer_qa_node(
        {"thread_id": "search-test-thread", "messages": [HumanMessage(content="Find Python docs")]},
        CONFIG,
    )
    assert result["qa_answer"] == "Python reference: https://docs.python.org/3/"


@pytest.mark.parametrize("include_contents", [True, False])
async def test_configured_exa_is_preserved(monkeypatch, include_contents):
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    calls = []

    class Exa:
        def __init__(self, *, api_key):
            assert api_key == "test-exa-key"

        def search_and_contents(self, query, **kwargs):
            calls.append(("contents", query, kwargs))
            return "Exa contents"

        def search(self, query, **kwargs):
            calls.append(("search", query, kwargs))
            return "Exa links"

    monkeypatch.setattr(search, "Exa", Exa)
    result = await dispatch(
        {"query": "Python docs", "num_results": 2, "include_contents": include_contents}
    )
    assert result["success"] is True
    assert calls == [
        (
            "contents" if include_contents else "search",
            "Python docs",
            {"num_results": 2, "type": "auto", **({"text": True} if include_contents else {})},
        )
    ]


async def test_explicit_parallel_overrides_exa(monkeypatch, mcp_server):
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")
    monkeypatch.setenv("PARALLEL_API_KEY", "ambient-parallel-key")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "parallel")
    assert (await dispatch())["success"] is True


@pytest.mark.parametrize(
    "provider,error",
    [
        ("exa", "EXA_API_KEY"),
        ("disabled", "disabled"),
        ("typo", "WEB_SEARCH_PROVIDER"),
    ],
)
async def test_explicit_choice_never_falls_back(monkeypatch, mcp_server, provider, error):
    requests, _ = mcp_server
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", provider)
    result = await dispatch()
    assert result["success"] is False
    assert error in result["error"]
    assert requests == []


async def test_exa_error_does_not_fall_back(monkeypatch, mcp_server):
    requests, _ = mcp_server
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")

    def broken_exa(**kwargs):
        raise RuntimeError("Exa unavailable")

    monkeypatch.setattr(search, "Exa", broken_exa)
    assert "Exa unavailable" in (await dispatch())["error"]
    assert requests == []


@pytest.mark.parametrize("failure", ["tool", "invalid", "http"])
async def test_mcp_errors_are_tool_results(mcp_server, failure):
    _, state = mcp_server
    if failure == "tool":
        state.update(error=True, result={"error": "quota exceeded"})
    elif failure == "invalid":
        state["result"] = {"unexpected": []}
    else:
        state["http_error"] = 503
    result = await dispatch()
    assert result["success"] is False
    assert result["results"] is None
    assert result["error"]


async def test_timeout_returns_failure(monkeypatch, mcp_server):
    _, state = mcp_server
    state["delay"] = True
    monkeypatch.setattr(parallel, "SEARCH_TIMEOUT", 0.05)
    result = await dispatch()
    assert result["success"] is False
    assert "Timeout" in result["error"]


async def test_cancellation_propagates(mcp_server):
    requests, state = mcp_server
    state["delay"] = True
    task = asyncio.create_task(dispatch())
    for _ in range(200):
        if any(json.loads(r.content)["method"] == "tools/call" for r in requests):
            break
        await asyncio.sleep(0.001)
    else:
        pytest.fail("MCP call did not start")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)


async def test_tool_exclusion_preserves_permissions(mcp_server):
    requests, _ = mcp_server
    node = dispatcher([])
    result = await node.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": {"query": "Python docs"},
                            "id": "search-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
    )
    assert result["messages"][-1].status == "error"
    assert requests == []
