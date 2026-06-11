from __future__ import annotations

import json
from uuid import UUID

import asyncpg
import httpx
import pytest
from harness.types import CallCtx, Principal, ToolCall
from respx import MockRouter


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


def _ctx() -> CallCtx:
    return CallCtx(
        tenant_id="tenant-a",
        principal=Principal(user_id="user-a"),
        run_id=UUID(int=1400),
        node_id="node-a",
        root_run_id=UUID(int=1400),
    )


@pytest.mark.asyncio
async def test_mcp_ingest_creates_registry_rows_and_reingest_upserts(
    pg: str,
    respx_mock: MockRouter,
) -> None:
    from harness.tools.mcp_adapter import MCPAdapter
    from harness.tools.registry import ToolRegistry

    route = respx_mock.post("https://mcp.test/mcp").mock(
        return_value=httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "list-tools",
                "result": {
                    "tools": [
                        {
                            "name": "calendar.lookup",
                            "description": "Find calendar events. Extra sentence.",
                            "inputSchema": {"type": "object"},
                            "annotations": {"freshness": "session"},
                        }
                    ]
                },
            },
        )
    )
    await _reset_and_migrate(pg)
    adapter = MCPAdapter("https://mcp.test/mcp")
    registry = ToolRegistry(pg, executors={adapter.source: adapter.invoke})

    await adapter.ingest(registry, tenant_id="tenant-a")
    await adapter.ingest(registry, tenant_id="tenant-a")

    tools = await registry.list_tools("tenant-a")

    assert route.call_count == 2
    assert len(tools) == 1
    assert tools[0].tool_id == "calendar.lookup"
    assert tools[0].source == adapter.source
    assert tools[0].side_effect == "read"
    assert tools[0].freshness == "session"
    assert tools[0].index_card == "calendar.lookup - Find calendar events."


@pytest.mark.asyncio
async def test_mcp_adapter_invokes_tool_through_registry(
    pg: str,
    respx_mock: MockRouter,
) -> None:
    from harness.tools.mcp_adapter import MCPAdapter
    from harness.tools.registry import ToolRegistry

    calls: list[dict[str, object]] = []

    def responder(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        decoded = json.loads(payload)
        calls.append(decoded)
        if decoded["method"] == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": decoded["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "search",
                                "description": "Search places.",
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": decoded["id"],
                "result": {"content": [{"type": "text", "text": "Roma Antica"}]},
            },
        )

    respx_mock.post("https://mcp.test/mcp").mock(side_effect=responder)
    await _reset_and_migrate(pg)
    adapter = MCPAdapter("https://mcp.test/mcp")
    registry = ToolRegistry(pg, executors={adapter.source: adapter.invoke})
    await adapter.ingest(registry, tenant_id="tenant-a")

    result = await registry.invoke(ToolCall(tool_id="search", args={"q": "pizza"}), _ctx())

    assert result.output == {"content": [{"type": "text", "text": "Roma Antica"}]}
    assert calls[-1]["method"] == "tools/call"
    assert calls[-1]["params"] == {"name": "search", "arguments": {"q": "pizza"}}
