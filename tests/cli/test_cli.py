from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg
from typer.testing import CliRunner


async def _reset_public_schema(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()


def test_cli_init_scaffolds_project(tmp_path: Path) -> None:
    from harness.cli.__main__ import app

    target = tmp_path / "demo"
    result = CliRunner().invoke(app, ["init", str(target)])

    assert result.exit_code == 0
    assert (target / "agent.yaml").exists()
    assert "created" in result.output


def test_cli_dev_reports_health() -> None:
    from harness.cli.__main__ import app

    result = CliRunner().invoke(app, ["dev", "--check-only"])

    assert result.exit_code == 0
    assert "/healthz ok" in result.output


def test_cli_run_executes_yaml_spec_with_stdin_input(tmp_path: Path, pg: str) -> None:
    from harness.cli.__main__ import app
    from harness.dsl.models import dumps_yaml

    from tests.engine.test_executor_basic import _spec

    spec_path = tmp_path / "agent.yaml"
    spec_path.write_text(dumps_yaml(_spec([("a", "b")])), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["run", str(spec_path), "--dsn", pg],
        input='{"goal":"demo"}',
    )

    assert result.exit_code == 0
    assert "succeeded" in result.output
    assert "terminal_node" in result.output


def test_cli_migrate_and_replay(pg: str) -> None:
    from harness.cli.__main__ import app
    from harness.durability.postgres.backend import PostgresBackend
    from harness.types import Principal, RunInit

    asyncio.run(_reset_public_schema(pg))
    runner = CliRunner()
    migrated = runner.invoke(app, ["migrate", "--dsn", pg])
    assert migrated.exit_code == 0

    backend = PostgresBackend(pg, background_flush=False)
    run_id = "00000000-0000-0000-0000-000000001717"
    asyncio.run(
        backend.create_run(
            RunInit(
                run_id=run_id,
                root_run_id=run_id,
                tenant_id="tenant-a",
                principal=Principal(user_id="user-a"),
                spec_id="cli-demo",
                spec_version=1,
                request_class="interactive",
                status="succeeded",
                budget={},
                result={"ok": True},
            )
        )
    )

    replayed = runner.invoke(app, ["replay", run_id, "--dsn", pg])

    assert replayed.exit_code == 0
    assert '"status": "succeeded"' in replayed.output
