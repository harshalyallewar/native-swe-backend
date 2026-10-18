# Web search

The QA and implementation agents use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
for web search when `EXA_API_KEY` and `WEB_SEARCH_PROVIDER` are empty. No Parallel account or API key
is needed. Queries are sent to Parallel; the service has free-tier rate limits. Search returns source
URLs and relevant excerpts, rather than full pages. `num_results` limits the returned sources, and
`include_contents=False` omits excerpts. The existing `fetch_url` tool can still read a specific page.

Set `WEB_SEARCH_PROVIDER=exa` to require Exa, `parallel` to select Parallel even when Exa is configured,
or `disabled` to prevent the web search tool from making requests. Provider errors are returned to
the agent without switching services. Parallel requests have a 30-second timeout and respect agent
cancellation. Restart the server after changing environment settings.

Environment settings:

```dotenv
WEB_SEARCH_PROVIDER=
EXA_API_KEY=
```

With no provider override, a nonempty `EXA_API_KEY` keeps Exa selected. Parallel never reads
`PARALLEL_API_KEY`; its built-in MCP connection is anonymous. Use Exa when full-page search
contents are required. Parallel may return fewer sources than requested.
