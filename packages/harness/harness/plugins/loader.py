"""Plugin provider loader and run session lifecycle."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from harness.tools.registry import ToolExecutor, ToolRegistry
from harness.types import CallCtx, ToolCall, ToolRecord, ToolResult


class PluginTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_id: str
    name: str
    description: str
    input_schema: dict[str, object]
    side_effect: str = "read"
    idempotency: str = "none"
    freshness: str = "volatile"
    auth_mode: str = "service"
    requires_approval: bool = False


class PluginManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    tools: list[PluginTool] = Field(default_factory=list)


class PluginProvider(Protocol):
    def manifest(self) -> PluginManifest: ...
    async def open_session(self, ctx: CallCtx) -> object | None: ...
    async def invoke(self, call: ToolCall, ctx: CallCtx, session: object | None) -> ToolResult: ...
    async def close_session(self, session: object) -> None: ...


class PluginManager:
    """Install plugin tools into the registry and manage per-run sessions."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        providers: Sequence[PluginProvider] | None = None,
    ) -> None:
        self._registry = registry
        self._providers = list(providers) if providers is not None else discover_plugins()
        self._sessions: dict[tuple[str, UUID], object | None] = {}

    async def install(self, *, tenant_id: str) -> None:
        for provider in self._providers:
            manifest = provider.manifest()
            source = _source(manifest.provider_id)
            self._registry.register_executor(source, self._executor(provider, manifest.provider_id))
            for tool in manifest.tools:
                await self._registry.upsert_tool(_record(tenant_id, source, tool))

    async def close_run(self, run_id: UUID) -> None:
        for provider in self._providers:
            manifest = provider.manifest()
            key = (manifest.provider_id, run_id)
            if key not in self._sessions:
                continue
            session = self._sessions.pop(key)
            if session is not None:
                await provider.close_session(session)

    def _executor(self, provider: PluginProvider, provider_id: str) -> ToolExecutor:
        async def invoke(call: ToolCall, ctx: CallCtx, _headers: dict[str, str]) -> ToolResult:
            key = (provider_id, ctx.run_id)
            session = self._sessions.get(key)
            if key not in self._sessions:
                session = await provider.open_session(ctx)
                self._sessions[key] = session
            try:
                return await provider.invoke(call, ctx, session)
            except Exception:
                if key in self._sessions:
                    session = self._sessions.pop(key)
                    if session is not None:
                        await provider.close_session(session)
                raise

        return invoke


def discover_plugins() -> list[PluginProvider]:
    discovered: list[PluginProvider] = []
    for entry_point in entry_points(group="harness.plugins"):
        provider = entry_point.load()
        discovered.append(provider() if isinstance(provider, type) else provider)
    return discovered


def _record(tenant_id: str, source: str, tool: PluginTool) -> ToolRecord:
    return ToolRecord(
        tenant_id=tenant_id,
        tool_id=tool.tool_id,
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        source=source,
        side_effect=_side_effect(tool.side_effect),
        idempotency=_idempotency(tool.idempotency),
        freshness=_freshness(tool.freshness),
        auth_mode=_auth_mode(tool.auth_mode),
        requires_approval=tool.requires_approval,
        index_card=_index_card(tool.tool_id, tool.description),
    )


def _source(provider_id: str) -> str:
    return f"plugin:{provider_id}"


def _index_card(tool_id: str, description: str) -> str:
    first_sentence = description.split(".", 1)[0].strip()
    return f"{tool_id} - {first_sentence}."


def _side_effect(value: str) -> Literal["pure", "read", "write"]:
    if value == "pure":
        return "pure"
    if value == "write":
        return "write"
    if value == "read":
        return "read"
    return "read"


def _idempotency(value: str) -> Literal["keyed", "none"]:
    if value == "keyed":
        return "keyed"
    if value == "none":
        return "none"
    return "none"


def _freshness(value: str) -> Literal["pure", "session", "volatile"]:
    if value == "pure":
        return "pure"
    if value == "session":
        return "session"
    if value == "volatile":
        return "volatile"
    return "volatile"


def _auth_mode(value: str) -> Literal["service", "user_passthrough"]:
    if value == "service":
        return "service"
    if value == "user_passthrough":
        return "user_passthrough"
    return "service"
