"""Cassette record/replay store for LLM and tool calls."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, cast

import asyncpg

from harness.errors import MissingCassette
from harness.types import LLMEvent, ToolResult
from harness.util import canon_hash

CassetteKind = Literal["llm", "tool"]
CassetteMode = Literal["record", "replay", "hybrid", "live"]


class CassetteStore:
    def __init__(self, dsn: str, *, pool: asyncpg.Pool | None = None) -> None:
        self._dsn = dsn
        self._pool = pool

    async def put(self, kind: CassetteKind, key: str, payload: object) -> None:
        async with self._connection() as conn:
            await conn.execute(
                """
                INSERT INTO cassettes (cassette_type, cache_key, payload)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (cassette_type, cache_key) DO UPDATE
                SET payload = EXCLUDED.payload,
                    created_at = now()
                """,
                kind,
                key,
                json.dumps(payload),
            )

    async def get(self, kind: CassetteKind, key: str) -> object | None:
        async with self._connection() as conn:
            value = cast(
                object | None,
                await conn.fetchval(
                """
                SELECT payload
                FROM cassettes
                WHERE cassette_type = $1 AND cache_key = $2
                """,
                kind,
                key,
                ),
            )
        if isinstance(value, str):
            return cast(object, json.loads(value))
        return value

    async def export_jsonl(self, path: Path) -> None:
        async with self._connection() as conn:
            rows = await conn.fetch(
                """
                SELECT cassette_type, cache_key, payload
                FROM cassettes
                ORDER BY cassette_type ASC, cache_key ASC
                """
            )
        lines = [
            json.dumps(
                {
                    "type": row["cassette_type"],
                    "key": row["cache_key"],
                    "payload": _json_value(row["payload"]),
                },
                sort_keys=True,
            )
            for row in rows
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    async def import_jsonl(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            await self.put(row["type"], row["key"], row["payload"])

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[asyncpg.Connection]:
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                yield conn
            return

        conn = await asyncpg.connect(self._dsn)
        try:
            yield conn
        finally:
            await conn.close()


class CassetteSession:
    def __init__(self, store: CassetteStore, *, mode: CassetteMode) -> None:
        self._store = store
        self._mode = mode

    async def llm(
        self,
        key_parts: object,
        live: Callable[[], AsyncIterator[LLMEvent]],
    ) -> AsyncIterator[LLMEvent]:
        key = canon_hash(key_parts)
        if self._mode in {"replay", "hybrid"}:
            payload = await self._store.get("llm", key)
            if payload is not None:
                for event in _events_from_payload(payload):
                    yield event
                return
            if self._mode == "replay":
                raise MissingCassette(f"missing llm cassette: {key}")

        events: list[LLMEvent] = []
        async for event in live():
            events.append(event)
            yield event
        if self._mode in {"record", "hybrid"}:
            await self._store.put("llm", key, [event.model_dump(mode="json") for event in events])

    async def tool(
        self,
        key_parts: object,
        live: Callable[[], Awaitable[ToolResult]],
    ) -> ToolResult:
        key = canon_hash(key_parts)
        if self._mode in {"replay", "hybrid"}:
            payload = await self._store.get("tool", key)
            if payload is not None:
                return ToolResult.model_validate(payload)
            if self._mode == "replay":
                raise MissingCassette(f"missing tool cassette: {key}")

        result = await live()
        if self._mode in {"record", "hybrid"}:
            await self._store.put("tool", key, result.model_dump(mode="json"))
        return result


def _events_from_payload(payload: object) -> list[LLMEvent]:
    if not isinstance(payload, list):
        raise MissingCassette("llm cassette payload is not an event list")
    return [LLMEvent.model_validate(event) for event in payload]


def _json_value(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value
