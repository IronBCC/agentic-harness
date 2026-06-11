"""Postgres node task queue."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import cast
from uuid import UUID

import asyncpg

from harness.types import NodeTask


class Queue:
    """Claim and manage node task leases using Postgres row locks."""

    def __init__(
        self,
        dsn: str,
        *,
        lease_seconds: int = 30,
        pool: asyncpg.Pool | None = None,
    ) -> None:
        self._dsn = dsn
        self._lease_seconds = lease_seconds
        self._pool = pool

    async def claim(self, worker: str, n: int) -> list[NodeTask]:
        """Claim up to n pending tasks without double delivery."""
        if n <= 0:
            return []
        async with self._connection() as conn:
            rows = await conn.fetch(
                """
                WITH picked AS (
                  SELECT task_id
                  FROM node_tasks
                  WHERE state = 'pending'
                    AND available_at <= now()
                  ORDER BY priority DESC, available_at ASC, created_at ASC
                  FOR UPDATE SKIP LOCKED
                  LIMIT $2
                )
                UPDATE node_tasks AS t
                SET state = 'claimed',
                    lease_owner = $1,
                    lease_expires_at = now() + ($3 * interval '1 second')
                FROM picked
                WHERE t.task_id = picked.task_id
                RETURNING t.task_id, t.run_id, t.node_id, t.attempt, t.input
                """,
                worker,
                n,
                self._lease_seconds,
            )
            return [self._task_from_row(row) for row in rows]

    async def heartbeat(self, worker: str, task_ids: list[UUID]) -> None:
        """Extend leases for tasks still owned by worker."""
        if not task_ids:
            return
        async with self._connection() as conn:
            await conn.execute(
                """
                UPDATE node_tasks
                SET lease_expires_at = now() + ($3 * interval '1 second')
                WHERE lease_owner = $1
                  AND task_id = ANY($2::uuid[])
                  AND state = 'claimed'
                """,
                worker,
                task_ids,
                self._lease_seconds,
            )

    async def complete(self, task_id: UUID, terminal: object) -> None:
        """Mark a claimed task done."""
        async with self._connection() as conn:
            await conn.execute(
                """
                UPDATE node_tasks
                SET state = 'done',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    input = coalesce(input, '{}'::jsonb) || $2::jsonb
                WHERE task_id = $1
                """,
                task_id,
                json.dumps({"terminal": terminal}),
            )

    async def reschedule(self, task_id: UUID, at: datetime, attempt: int) -> None:
        """Return a task to pending at a future availability time."""
        async with self._connection() as conn:
            await conn.execute(
                """
                UPDATE node_tasks
                SET state = 'pending',
                    attempt = $2,
                    available_at = $3,
                    lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE task_id = $1
                """,
                task_id,
                attempt,
                at,
            )
            await conn.execute("SELECT pg_notify('tasks', $1)", str(task_id))

    async def reap_expired(self) -> list[NodeTask]:
        """Return expired claimed tasks to pending and return their task snapshots."""
        async with self._connection() as conn:
            rows = await conn.fetch(
                """
                WITH expired AS (
                  SELECT task_id
                  FROM node_tasks
                  WHERE state = 'claimed'
                    AND lease_expires_at IS NOT NULL
                    AND lease_expires_at <= now()
                  ORDER BY lease_expires_at ASC
                  FOR UPDATE SKIP LOCKED
                )
                UPDATE node_tasks AS t
                SET state = 'pending',
                    lease_owner = NULL,
                    lease_expires_at = NULL
                FROM expired
                WHERE t.task_id = expired.task_id
                RETURNING t.task_id, t.run_id, t.node_id, t.attempt, t.input
                """
            )
            if rows:
                await conn.execute("SELECT pg_notify('tasks', 'reap_expired')")
            return [self._task_from_row(row) for row in rows]

    async def lease_expires_at(self, task_id: UUID) -> datetime | None:
        """Return a task's lease expiry timestamp."""
        async with self._connection() as conn:
            value = await conn.fetchval(
                "SELECT lease_expires_at FROM node_tasks WHERE task_id = $1",
                task_id,
            )
            return cast(datetime | None, value)

    async def task_state(self, task_id: UUID) -> str | None:
        """Return a task's state."""
        async with self._connection() as conn:
            value = await conn.fetchval(
                "SELECT state FROM node_tasks WHERE task_id = $1",
                task_id,
            )
            return cast(str | None, value)

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

    def _task_from_row(self, row: asyncpg.Record) -> NodeTask:
        raw_input = row["input"]
        if isinstance(raw_input, str):
            task_input = json.loads(raw_input)
        elif isinstance(raw_input, dict):
            task_input = raw_input
        elif raw_input is None:
            task_input = {}
        else:
            task_input = dict(raw_input)
        return NodeTask(
            task_id=row["task_id"],
            run_id=row["run_id"],
            node_id=row["node_id"],
            attempt=row["attempt"],
            input=task_input,
        )
