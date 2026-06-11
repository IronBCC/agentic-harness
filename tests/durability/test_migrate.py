from __future__ import annotations

import asyncpg
import pytest


async def _reset_public_schema(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()


async def _table_exists(conn: asyncpg.Connection, table_name: str) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.tables
              WHERE table_schema = 'public'
                AND table_name = $1
            )
            """,
            table_name,
        )
    )


async def _constraint_exists(conn: asyncpg.Connection, table: str, columns: list[str]) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
              SELECT 1
              FROM pg_constraint c
              JOIN pg_class t ON t.oid = c.conrelid
              JOIN pg_namespace n ON n.oid = t.relnamespace
              WHERE n.nspname = 'public'
                AND t.relname = $1
                AND c.contype = 'u'
                AND (
                  SELECT array_agg(a.attname::text ORDER BY ck.ord)
                  FROM unnest(c.conkey) WITH ORDINALITY AS ck(attnum, ord)
                  JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ck.attnum
                ) = $2::text[]
            )
            """,
            table,
            columns,
        )
    )


@pytest.mark.asyncio
async def test_apply_creates_core_schema_and_is_idempotent(pg: str) -> None:
    from harness.durability.postgres.migrate import apply

    await _reset_public_schema(pg)

    first = await apply(pg)
    second = await apply(pg)

    conn = await asyncpg.connect(pg)
    try:
        assert first == [1]
        assert second == []
        for table in ("schema_migrations", "runs", "run_events", "node_tasks"):
            assert await _table_exists(conn, table)
        assert await _constraint_exists(conn, "run_events", ["run_id", "idempotency_key"])
        assert await _constraint_exists(conn, "run_events", ["run_id", "seq"])
    finally:
        await conn.close()
