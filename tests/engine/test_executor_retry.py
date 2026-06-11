from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest
from harness.dsl.models import AgentSpec, NodeSpec
from harness.errors import InfraRetryable, SchemaViolation
from harness.types import NodeTask, NodeYield, Principal, RunInit
from harness.util import FrozenClock, SequentialIdGen


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


def _run_init(run_id: UUID) -> RunInit:
    return RunInit(
        run_id=run_id,
        root_run_id=run_id,
        tenant_id="tenant-a",
        principal=Principal(user_id="user-a"),
        spec_id="retry-demo",
        spec_version=1,
        request_class="interactive",
        budget={},
    )


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "spec_id": "retry-demo",
            "version": 1,
            "description": "Retry demo",
            "fact_types": [{"name": "result", "schema": {"type": "object"}}],
            "tools": [],
            "models": [
                {"name": "gemma4-nano", "provider": "fake", "model": "nano"},
                {"name": "gemma4", "provider": "fake", "model": "full"},
            ],
            "nodes": [
                {
                    "node_id": "triage",
                    "kind": "leaf",
                    "model": "gemma4-nano",
                    "escalation": ["gemma4"],
                    "max_context_tokens": 1024,
                    "prompt_template": "triage",
                    "retry": {"max_attempts": 3, "backoff_ms": 250, "jitter": 0.0},
                    "cost_estimate": {
                        "tokens": 1,
                        "usd": 0.0,
                        "wall_ms": 1,
                        "llm_calls": 1,
                    },
                }
            ],
            "edges": [],
            "budget_policy": {
                "pools": {
                    "interactive": {
                        "tokens": 100,
                        "usd": 1.0,
                        "wall_ms": 1000,
                        "llm_calls": 5,
                    }
                },
                "degradation": {},
            },
            "policies": {},
            "evals": [{"case_id": "smoke", "input": {}, "expected": {}}],
        }
    )


@pytest.mark.asyncio
async def test_executor_reschedules_infra_retryable_with_backoff(pg: str) -> None:
    from harness.durability.postgres.backend import PostgresBackend
    from harness.engine.executor import Executor

    run_id = UUID(int=900)
    now = datetime(2030, 1, 1, 9, 0, tzinfo=UTC)
    await _reset_and_migrate(pg)
    backend = PostgresBackend(pg, background_flush=False)

    async def runner(_node: NodeSpec, _task: NodeTask) -> NodeYield:
        raise InfraRetryable("timeout")

    executor = Executor(
        backend=backend,
        spec=_spec(),
        node_runner=runner,
        idgen=SequentialIdGen(900_000),
        worker="retry-worker",
        clock=FrozenClock(now),
    )

    await executor.seed_run(_run_init(run_id), input={})
    assert await executor.run_once() == 1

    tasks = await backend.list_tasks(run_id)
    loaded = await backend.load(run_id)

    assert [(task.node_id, task.state, task.attempt) for task in tasks] == [
        ("triage", "pending", 1)
    ]
    assert tasks[0].lease_owner is None
    assert tasks[0].lease_expires_at is None
    assert await backend.claim("too-early", 1) == []
    assert loaded.status == "running"
    assert [event.kind.value for event in loaded.events] == ["node_started"]
    assert tasks[0].input == {}

    conn = await asyncpg.connect(pg)
    try:
        available_at = await conn.fetchval(
            "SELECT available_at FROM node_tasks WHERE task_id = $1",
            tasks[0].task_id,
        )
    finally:
        await conn.close()
    assert available_at == now + timedelta(milliseconds=250)


@pytest.mark.asyncio
async def test_executor_schema_violation_reschedule_carries_escalation_state(pg: str) -> None:
    from harness.durability.postgres.backend import PostgresBackend
    from harness.engine.executor import Executor

    run_id = UUID(int=901)
    now = datetime(2030, 1, 1, 9, 0, tzinfo=UTC)
    await _reset_and_migrate(pg)
    backend = PostgresBackend(pg, background_flush=False)

    async def runner(_node: NodeSpec, _task: NodeTask) -> NodeYield:
        raise SchemaViolation("bad json")

    executor = Executor(
        backend=backend,
        spec=_spec(),
        node_runner=runner,
        idgen=SequentialIdGen(901_000),
        worker="schema-worker",
        clock=FrozenClock(now),
    )

    await executor.seed_run(_run_init(run_id), input={})
    await executor.run_once()
    tasks_after_first = await backend.list_tasks(run_id)
    assert tasks_after_first[0].input == {"model_index": 0, "schema_retries": 1}

    force_available = datetime(2000, 1, 1, tzinfo=UTC)
    await executor.backend.reschedule(
        tasks_after_first[0].task_id,
        force_available,
        tasks_after_first[0].attempt,
    )
    await executor.run_once()
    tasks_after_second = await backend.list_tasks(run_id)
    assert tasks_after_second[0].input == {"model_index": 0, "schema_retries": 2}

    await executor.backend.reschedule(
        tasks_after_second[0].task_id,
        force_available,
        tasks_after_second[0].attempt,
    )
    await executor.run_once()
    tasks_after_escalation = await backend.list_tasks(run_id)
    assert tasks_after_escalation[0].input == {"model_index": 1, "schema_retries": 0}
