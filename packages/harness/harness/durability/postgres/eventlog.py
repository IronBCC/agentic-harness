"""Postgres-backed run event log."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg

from harness.errors import FencingError
from harness.types import Event, EventKind


class EventLog:
    """Append-only event log with per-run sequence fencing."""

    def __init__(
        self,
        dsn: str,
        *,
        flush_interval_s: float = 0.1,
        background_flush: bool = True,
        pool: asyncpg.Pool | None = None,
    ) -> None:
        self._dsn = dsn
        self._flush_interval_s = flush_interval_s
        self._background_flush = background_flush
        self._pool = pool
        self._buffer: dict[UUID, list[Event]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None

    async def append(self, run_id: UUID, events: list[Event]) -> None:
        """Buffer a batch and flush synchronously when any event is a barrier.

        Callers own sequence assignment. A duplicate per-run sequence means a
        stale owner raced a lease takeover, so the write is fenced.
        """
        if not events:
            return
        async with self._lock:
            self._buffer[run_id].extend(events)
            self._ensure_background_flush()
        if any(event.barrier for event in events):
            await self.flush()

    async def flush(self) -> None:
        """Commit all buffered events in one transaction per run."""
        async with self._lock:
            batches = dict(self._buffer)
            self._buffer.clear()
        try:
            for run_id, events in batches.items():
                await self._write_batch(run_id, events)
        except Exception:
            async with self._lock:
                for run_id, events in batches.items():
                    self._buffer[run_id] = events + self._buffer[run_id]
            raise

    async def close(self) -> None:
        """Flush buffered events and stop the background flush task."""
        await self.flush()
        if self._flush_task is None:
            return
        self._flush_task.cancel()
        try:
            await self._flush_task
        except asyncio.CancelledError:
            pass
        self._flush_task = None

    def _ensure_background_flush(self) -> None:
        if not self._background_flush:
            return
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval_s)
            await self.flush()

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

    async def _write_batch(self, run_id: UUID, events: Iterable[Event]) -> None:
        try:
            async with self._connection() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        """
                        INSERT INTO run_events (
                          run_id, seq, node_id, kind, payload, idempotency_key
                        )
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        tuple(
                            (
                                run_id,
                                event.seq,
                                event.node_id,
                                event.kind.value,
                                json.dumps(event.payload),
                                event.idempotency_key,
                            )
                            for event in events
                        ),
                    )
        except asyncpg.UniqueViolationError as exc:
            raise FencingError("duplicate event sequence or idempotency key") from exc

    async def load(self, run_id: UUID) -> list[Event]:
        """Load a run's events in replay order."""
        async with self._connection() as conn:
            rows = await conn.fetch(
                """
                SELECT seq, node_id, kind, payload, idempotency_key
                FROM run_events
                WHERE run_id = $1
                ORDER BY seq ASC
                """,
                run_id,
            )

        return [
            Event(
                seq=row["seq"],
                node_id=row["node_id"],
                kind=EventKind(row["kind"]),
                payload=json.loads(row["payload"]),
                idempotency_key=row["idempotency_key"],
            )
            for row in rows
        ]
