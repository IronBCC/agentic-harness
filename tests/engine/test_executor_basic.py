from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from uuid import UUID

import asyncpg
import pytest
from harness.dsl.models import AgentSpec, NodeSpec
from harness.types import NodeTask, NodeYield, Principal, RunInit, YieldStatus
from harness.util import SequentialIdGen

Runner = Callable[[NodeSpec, NodeTask], Awaitable[NodeYield]]


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
        spec_id="executor-demo",
        spec_version=1,
        request_class="interactive",
        budget={"tokens": 1000, "usd": 1.0},
    )


def _spec(edges: list[tuple[str, str]]) -> AgentSpec:
    node_ids = sorted({node_id for edge in edges for node_id in edge})
    if not node_ids:
        node_ids = ["a"]
    return AgentSpec.model_validate(
        {
            "spec_id": "executor-demo",
            "version": 1,
            "description": "Executor test graph",
            "fact_types": [{"name": "result", "schema": {"type": "object"}}],
            "tools": [],
            "models": [{"name": "fake", "provider": "fake", "model": "fake"}],
            "nodes": [
                {
                    "node_id": node_id,
                    "kind": "leaf",
                    "model": "fake",
                    "max_context_tokens": 1024,
                    "prompt_template": f"node.{node_id}",
                    "retry": {"max_attempts": 0, "backoff_ms": 0, "jitter": 0.0},
                    "cost_estimate": {
                        "tokens": 1,
                        "usd": 0.0,
                        "wall_ms": 1,
                        "llm_calls": 0,
                    },
                }
                for node_id in node_ids
            ],
            "edges": [{"from_node": src, "to_node": dst} for src, dst in edges],
            "budget_policy": {
                "pools": {
                    "interactive": {
                        "tokens": 100,
                        "usd": 1.0,
                        "wall_ms": 10000,
                        "llm_calls": 10,
                    }
                },
                "degradation": {},
            },
            "policies": {},
            "evals": [{"case_id": "smoke", "input": {}, "expected": {}}],
        }
    )


async def _done_runner(node: NodeSpec, _task: NodeTask) -> NodeYield:
    return NodeYield(
        status=YieldStatus.done,
        result_ref=f"result:{node.node_id}",
        facts=[{"node": node.node_id}],
    )


@pytest.mark.asyncio
async def test_executor_completes_linear_graph(pg: str) -> None:
    from harness.durability.postgres.backend import PostgresBackend
    from harness.engine.executor import Executor

    run_id = UUID(int=400)
    await _reset_and_migrate(pg)
    backend = PostgresBackend(pg, background_flush=False)
    executor = Executor(
        backend=backend,
        spec=_spec([("a", "b"), ("b", "c")]),
        node_runner=_done_runner,
        idgen=SequentialIdGen(400_000),
        worker="linear-worker",
    )

    await executor.seed_run(_run_init(run_id), input={})
    await executor.run_until_idle()
    loaded = await backend.load(run_id)
    tasks = await backend.list_tasks(run_id)

    assert loaded.status == "succeeded"
    assert loaded.result == {"terminal_node": "c", "result_ref": "result:c"}
    assert [(task.node_id, task.state) for task in tasks] == [
        ("a", "done"),
        ("b", "done"),
        ("c", "done"),
    ]
    assert [event.kind.value for event in loaded.events] == [
        "node_started",
        "yield",
        "node_started",
        "yield",
        "node_started",
        "yield",
        "run_finished",
    ]


@pytest.mark.asyncio
async def test_executor_runs_ready_diamond_branches_concurrently(pg: str) -> None:
    from harness.durability.postgres.backend import PostgresBackend
    from harness.engine.executor import Executor

    run_id = UUID(int=401)
    await _reset_and_migrate(pg)
    backend = PostgresBackend(pg, background_flush=False)
    active = 0
    max_active = 0
    branch_started = 0
    both_branches_started = asyncio.Event()

    async def runner(node: NodeSpec, _task: NodeTask) -> NodeYield:
        nonlocal active, max_active, branch_started
        if node.node_id in {"b", "c"}:
            active += 1
            branch_started += 1
            max_active = max(max_active, active)
            if branch_started == 2:
                both_branches_started.set()
            await asyncio.wait_for(both_branches_started.wait(), timeout=1)
            await asyncio.sleep(0)
            active -= 1
        return NodeYield(status=YieldStatus.done, result_ref=f"result:{node.node_id}")

    executor = Executor(
        backend=backend,
        spec=_spec([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]),
        node_runner=runner,
        idgen=SequentialIdGen(401_000),
        worker="diamond-worker",
        batch_size=2,
    )

    await executor.seed_run(_run_init(run_id), input={})
    await executor.run_until_idle()
    loaded = await backend.load(run_id)

    assert loaded.status == "succeeded"
    assert max_active == 2


@pytest.mark.asyncio
async def test_executor_resumes_from_persisted_frontier_after_restart(pg: str) -> None:
    from harness.durability.postgres.backend import PostgresBackend
    from harness.engine.executor import Executor

    run_id = UUID(int=402)
    await _reset_and_migrate(pg)
    backend = PostgresBackend(pg, background_flush=False)
    spec = _spec([("a", "b"), ("b", "c")])
    first = Executor(
        backend=backend,
        spec=spec,
        node_runner=_done_runner,
        idgen=SequentialIdGen(402_000),
        worker="first-worker",
    )
    await first.seed_run(_run_init(run_id), input={})
    await first.run_once()

    after_first_batch = await backend.load(run_id)
    assert after_first_batch.status == "running"
    assert [(task.node_id, task.state) for task in await backend.list_tasks(run_id)] == [
        ("a", "done"),
        ("b", "pending"),
    ]

    second = Executor(
        backend=backend,
        spec=spec,
        node_runner=_done_runner,
        idgen=SequentialIdGen(403_000),
        worker="second-worker",
    )
    await second.run_until_idle()
    loaded = await backend.load(run_id)

    assert loaded.status == "succeeded"
    assert [(task.node_id, task.state) for task in await backend.list_tasks(run_id)] == [
        ("a", "done"),
        ("b", "done"),
        ("c", "done"),
    ]
