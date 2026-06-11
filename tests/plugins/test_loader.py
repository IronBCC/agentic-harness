from __future__ import annotations

from uuid import UUID

import asyncpg
import pytest
from harness.types import CallCtx, Principal, ToolCall, ToolResult

RUN_ID = UUID(int=1600)


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


def _ctx(run_id: UUID = RUN_ID) -> CallCtx:
    return CallCtx(
        tenant_id="tenant-a",
        principal=Principal(user_id="user-a"),
        run_id=run_id,
        node_id="node-a",
        root_run_id=run_id,
    )


class EchoProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.opens = 0
        self.closes = 0
        self.invocations = 0

    def manifest(self) -> object:
        from harness.plugins.loader import PluginManifest, PluginTool

        return PluginManifest(
            provider_id="echo",
            tools=[
                PluginTool(
                    tool_id="echo.say",
                    name="Echo Say",
                    description="Echo input text.",
                    input_schema={"type": "object"},
                    freshness="volatile",
                )
            ],
        )

    async def open_session(self, ctx: CallCtx) -> dict[str, object]:
        self.opens += 1
        return {"run_id": str(ctx.run_id)}

    async def invoke(
        self,
        call: ToolCall,
        _ctx: CallCtx,
        session: object,
    ) -> ToolResult:
        self.invocations += 1
        if self.fail:
            raise RuntimeError("boom")
        return ToolResult(output={"echo": call.args["text"], "session": session})

    async def close_session(self, session: object) -> None:
        assert isinstance(session, dict)
        self.closes += 1


@pytest.mark.asyncio
async def test_plugin_provider_appears_in_registry_and_invokes_through_pipeline(pg: str) -> None:
    from harness.plugins.loader import PluginManager
    from harness.tools.registry import ToolRegistry

    await _reset_and_migrate(pg)
    provider = EchoProvider()
    registry = ToolRegistry(pg, executors={})
    manager = PluginManager(registry, providers=[provider])
    await manager.install(tenant_id="tenant-a")

    tools = await registry.list_tools("tenant-a")
    result = await registry.invoke(ToolCall(tool_id="echo.say", args={"text": "hi"}), _ctx())

    assert [tool.tool_id for tool in tools] == ["echo.say"]
    assert tools[0].source == "plugin:echo"
    assert result.output == {"echo": "hi", "session": {"run_id": str(UUID(int=1600))}}


@pytest.mark.asyncio
async def test_plugin_session_opens_once_per_run_and_closes_on_terminal(pg: str) -> None:
    from harness.plugins.loader import PluginManager
    from harness.tools.registry import ToolRegistry

    await _reset_and_migrate(pg)
    provider = EchoProvider()
    registry = ToolRegistry(pg, executors={})
    manager = PluginManager(registry, providers=[provider])
    await manager.install(tenant_id="tenant-a")

    await registry.invoke(ToolCall(tool_id="echo.say", args={"text": "a"}), _ctx())
    await registry.invoke(ToolCall(tool_id="echo.say", args={"text": "b"}), _ctx())
    await manager.close_run(UUID(int=1600))

    assert provider.opens == 1
    assert provider.invocations == 2
    assert provider.closes == 1


@pytest.mark.asyncio
async def test_plugin_session_closes_on_failure(pg: str) -> None:
    from harness.plugins.loader import PluginManager
    from harness.tools.registry import ToolRegistry

    await _reset_and_migrate(pg)
    provider = EchoProvider(fail=True)
    registry = ToolRegistry(pg, executors={})
    manager = PluginManager(registry, providers=[provider])
    await manager.install(tenant_id="tenant-a")

    with pytest.raises(RuntimeError, match="boom"):
        await registry.invoke(ToolCall(tool_id="echo.say", args={"text": "x"}), _ctx())

    assert provider.opens == 1
    assert provider.closes == 1
