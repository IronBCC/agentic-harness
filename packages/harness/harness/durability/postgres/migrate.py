"""Plain SQL migration runner for the Postgres durability backend."""

from __future__ import annotations

from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).with_name("migrations")


async def apply(dsn: str) -> list[int]:
    """Apply unapplied migrations in order and return applied versions."""
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version    int PRIMARY KEY,
              applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        applied: list[int] = []
        for path in sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql")):
            version = int(path.name.split("_", 1)[0])
            already_applied = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE version = $1)",
                version,
            )
            if already_applied:
                continue
            sql = path.read_text(encoding="utf-8")
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES ($1)",
                    version,
                )
            applied.append(version)
        return applied
    finally:
        await conn.close()

