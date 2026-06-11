"""Tool registry CRUD and invoke pipeline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import asyncpg

from harness.errors import ToolOutcomeUnknown, ValidationFailed
from harness.types import CallCtx, ToolCall, ToolRecord, ToolResult
from harness.util import canon_hash

type ToolExecutor = Callable[[ToolCall, CallCtx, dict[str, str]], Awaitable[ToolResult]]


class AmbiguousToolError(Exception):
    """Transport failed after a non-idempotent write may have reached the tool."""


class ToolRegistry:
    """Postgres-backed tool registry with memoization and in-process singleflight."""

    def __init__(
        self,
        dsn: str,
        *,
        executors: dict[str, ToolExecutor],
        pool: asyncpg.Pool | None = None,
    ) -> None:
        self._dsn = dsn
        self._executors = executors
        self._pool = pool
        self._lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Task[ToolResult]] = {}

    async def upsert_tool(self, record: ToolRecord) -> None:
        async with self._connection() as conn:
            await conn.execute(
                """
                INSERT INTO tool_registry (
                  tenant_id, tool_id, name, description, input_schema, source,
                  side_effect, idempotency, freshness, auth_mode, requires_approval,
                  index_card, metadata
                )
                VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,$8,$9,$10,$11,$12,$13::jsonb)
                ON CONFLICT (tenant_id, tool_id) DO UPDATE
                SET name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    input_schema = EXCLUDED.input_schema,
                    source = EXCLUDED.source,
                    side_effect = EXCLUDED.side_effect,
                    idempotency = EXCLUDED.idempotency,
                    freshness = EXCLUDED.freshness,
                    auth_mode = EXCLUDED.auth_mode,
                    requires_approval = EXCLUDED.requires_approval,
                    index_card = EXCLUDED.index_card,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                record.tenant_id,
                record.tool_id,
                record.name,
                record.description,
                json.dumps(record.input_schema),
                record.source,
                record.side_effect,
                record.idempotency,
                record.freshness,
                record.auth_mode,
                record.requires_approval,
                record.index_card,
                json.dumps(record.metadata),
            )

    async def resolve(self, tool_id: str, tenant_id: str) -> ToolRecord:
        async with self._connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT tenant_id, tool_id, name, description, input_schema, source,
                       side_effect, idempotency, freshness, auth_mode, requires_approval,
                       index_card, metadata
                FROM tool_registry
                WHERE tenant_id = $1 AND tool_id = $2
                """,
                tenant_id,
                tool_id,
            )
        if row is None:
            raise ValidationFailed(f"tool not found: {tenant_id}/{tool_id}")
        return _record_from_row(row)

    async def invoke(self, call: ToolCall, ctx: CallCtx) -> ToolResult:
        record = await self.resolve(call.tool_id, ctx.tenant_id)
        if not _cacheable(record):
            return await self._execute(record, call, ctx)

        cache_key = _memo_key(record, call, ctx)
        cached = await self._load_memo(record, cache_key)
        if cached is not None:
            return cached

        async with self._lock:
            cached = await self._load_memo(record, cache_key)
            if cached is not None:
                return cached
            task = self._inflight.get(cache_key)
            owner = task is None
            if task is None:
                task = asyncio.create_task(self._execute_and_store(record, call, ctx, cache_key))
                self._inflight[cache_key] = task

        try:
            return await task
        finally:
            if owner:
                async with self._lock:
                    self._inflight.pop(cache_key, None)

    async def _execute_and_store(
        self,
        record: ToolRecord,
        call: ToolCall,
        ctx: CallCtx,
        cache_key: str,
    ) -> ToolResult:
        result = await self._execute(record, call, ctx)
        await self._store_memo(record, cache_key, result)
        return result

    async def _execute(self, record: ToolRecord, call: ToolCall, ctx: CallCtx) -> ToolResult:
        executor = self._executors.get(record.source)
        if executor is None:
            raise ValidationFailed(f"tool source executor not found: {record.source}")
        headers: dict[str, str] = {}
        if record.side_effect == "write" and record.idempotency == "keyed":
            headers["idempotency-key"] = canon_hash(
                {
                    "tenant_id": ctx.tenant_id,
                    "run_id": str(ctx.run_id),
                    "node_id": ctx.node_id,
                    "tool_id": call.tool_id,
                    "args": call.args,
                }
            )
        try:
            return await executor(call, ctx, headers)
        except AmbiguousToolError as exc:
            if record.side_effect == "write" and record.idempotency == "none":
                raise ToolOutcomeUnknown(str(exc)) from exc
            raise

    async def _load_memo(self, record: ToolRecord, cache_key: str) -> ToolResult | None:
        async with self._connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT output, artifact_hint
                FROM memo_cache
                WHERE tenant_id = $1 AND tool_id = $2 AND cache_key = $3
                """,
                record.tenant_id,
                record.tool_id,
                cache_key,
            )
        if row is None:
            return None
        return ToolResult(output=_json_dict(row["output"]), artifact_hint=row["artifact_hint"])

    async def _store_memo(self, record: ToolRecord, cache_key: str, result: ToolResult) -> None:
        async with self._connection() as conn:
            await conn.execute(
                """
                INSERT INTO memo_cache (tenant_id, tool_id, cache_key, output, artifact_hint)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                ON CONFLICT (tenant_id, tool_id, cache_key) DO UPDATE
                SET output = EXCLUDED.output,
                    artifact_hint = EXCLUDED.artifact_hint,
                    created_at = now()
                """,
                record.tenant_id,
                record.tool_id,
                cache_key,
                json.dumps(result.output),
                result.artifact_hint,
            )

    async def list_tools(self, tenant_id: str) -> list[ToolRecord]:
        async with self._connection() as conn:
            rows = await conn.fetch(
                """
                SELECT tenant_id, tool_id, name, description, input_schema, source,
                       side_effect, idempotency, freshness, auth_mode, requires_approval,
                       index_card, metadata
                FROM tool_registry
                WHERE tenant_id = $1
                ORDER BY tool_id ASC
                """,
                tenant_id,
            )
        return [_record_from_row(row) for row in rows]

    @asynccontextmanager
    async def _connection(self) -> AsyncConnection:
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                yield conn
            return

        conn = await asyncpg.connect(self._dsn)
        try:
            yield conn
        finally:
            await conn.close()


def _cacheable(record: ToolRecord) -> bool:
    return record.side_effect in {"pure", "read"} and record.freshness != "volatile"


def _memo_key(record: ToolRecord, call: ToolCall, ctx: CallCtx) -> str:
    return canon_hash(
        {
            "tenant_id": ctx.tenant_id,
            "principal": ctx.principal.model_dump(mode="json"),
            "tool_id": record.tool_id,
            "args": call.args,
        }
    )


def _record_from_row(row: asyncpg.Record) -> ToolRecord:
    return ToolRecord(
        tenant_id=row["tenant_id"],
        tool_id=row["tool_id"],
        name=row["name"],
        description=row["description"],
        input_schema=_json_dict(row["input_schema"]),
        source=row["source"],
        side_effect=row["side_effect"],
        idempotency=row["idempotency"],
        freshness=row["freshness"],
        auth_mode=row["auth_mode"],
        requires_approval=row["requires_approval"],
        index_card=row["index_card"],
        metadata=_json_dict(row["metadata"]),
    )


def _json_dict(value: object) -> dict[str, object]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return parsed if isinstance(parsed, dict) else {}


type AsyncConnection = asyncpg.Connection
