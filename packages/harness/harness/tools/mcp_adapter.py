"""MCP streamable-HTTP ingestion and invocation adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import httpx

from harness.tools.registry import ToolRegistry
from harness.types import CallCtx, ToolCall, ToolRecord, ToolResult


@dataclass(frozen=True)
class MCPAdapter:
    endpoint: str

    @property
    def source(self) -> str:
        return f"mcp:{self.endpoint}"

    async def ingest(self, registry: ToolRegistry, *, tenant_id: str) -> None:
        payload = await self._rpc("tools/list", {})
        result = payload.get("result")
        tools = result.get("tools", []) if isinstance(result, dict) else []
        for raw_tool in tools:
            if isinstance(raw_tool, dict):
                await registry.upsert_tool(_record_from_mcp(tenant_id, self.source, raw_tool))

    async def invoke(
        self,
        call: ToolCall,
        _ctx: CallCtx,
        _headers: dict[str, str],
    ) -> ToolResult:
        payload = await self._rpc(
            "tools/call",
            {"name": call.tool_id, "arguments": call.args},
        )
        result = payload.get("result")
        return ToolResult(output=result if isinstance(result, dict) else {})

    async def _rpc(self, method: str, params: dict[str, object]) -> dict[str, object]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.endpoint,
                json={"jsonrpc": "2.0", "id": method, "method": method, "params": params},
            )
            response.raise_for_status()
            parsed = response.json()
        return parsed if isinstance(parsed, dict) else {}


def _record_from_mcp(tenant_id: str, source: str, raw_tool: dict[str, object]) -> ToolRecord:
    name = str(raw_tool["name"])
    description = str(raw_tool.get("description", ""))
    annotations = raw_tool.get("annotations")
    annotations = annotations if isinstance(annotations, dict) else {}
    input_schema = raw_tool.get("inputSchema")
    input_schema = input_schema if isinstance(input_schema, dict) else {}
    return ToolRecord(
        tenant_id=tenant_id,
        tool_id=name,
        name=name,
        description=description,
        input_schema=input_schema,
        source=source,
        side_effect=_side_effect(annotations),
        idempotency=_idempotency(annotations),
        freshness=_freshness(annotations),
        auth_mode=_auth_mode(annotations),
        requires_approval=_bool_annotation(annotations, "requires_approval", False),
        index_card=_index_card(name, description),
        metadata={"mcp_endpoint": source.removeprefix("mcp:")},
    )


def _side_effect(values: dict[object, object]) -> Literal["pure", "read", "write"]:
    value = values.get("side_effect", "read")
    if value == "pure" or value == "read" or value == "write":
        return value
    return "read"


def _idempotency(values: dict[object, object]) -> Literal["keyed", "none"]:
    value = values.get("idempotency", "none")
    if value == "keyed" or value == "none":
        return value
    return "none"


def _freshness(values: dict[object, object]) -> Literal["pure", "session", "volatile"]:
    value = values.get("freshness", "volatile")
    if value == "pure" or value == "session" or value == "volatile":
        return value
    return "volatile"


def _auth_mode(values: dict[object, object]) -> Literal["service", "user_passthrough"]:
    value = values.get("auth_mode", "service")
    if value == "service" or value == "user_passthrough":
        return value
    return "service"


def _bool_annotation(values: dict[object, object], key: str, default: bool) -> bool:
    value = values.get(key, default)
    return value if isinstance(value, bool) else default


def _index_card(name: str, description: str) -> str:
    first_sentence = description.split(".", 1)[0].strip()
    card = f"{name} - {first_sentence}."
    words = card.split()
    if len(words) <= 15:
        return card
    return " ".join(words[:15])
