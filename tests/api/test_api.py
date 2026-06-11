from __future__ import annotations

from collections.abc import Awaitable, Callable

import asyncpg
import httpx
import pytest
from harness.dsl.models import AgentSpec, NodeSpec
from harness.types import NodeTask, NodeYield, YieldStatus
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


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "spec_id": "api-demo",
            "version": 1,
            "description": "API demo",
            "fact_types": [{"name": "result", "schema": {"type": "object"}}],
            "tools": [],
            "models": [{"name": "fake", "provider": "fake", "model": "fake"}],
            "nodes": [
                {
                    "node_id": "planner",
                    "kind": "planner",
                    "model": "fake",
                    "max_context_tokens": 1024,
                    "prompt_template": "plan",
                    "retry": {"max_attempts": 0, "backoff_ms": 0, "jitter": 0.0},
                    "cost_estimate": {
                        "tokens": 1,
                        "usd": 0.0,
                        "wall_ms": 1,
                        "llm_calls": 0,
                    },
                },
                {
                    "node_id": "synthesizer",
                    "kind": "synthesizer",
                    "model": "fake",
                    "max_context_tokens": 1024,
                    "prompt_template": "synthesize",
                    "retry": {"max_attempts": 0, "backoff_ms": 0, "jitter": 0.0},
                    "cost_estimate": {
                        "tokens": 1,
                        "usd": 0.0,
                        "wall_ms": 1,
                        "llm_calls": 0,
                    },
                },
            ],
            "edges": [{"from_node": "planner", "to_node": "synthesizer"}],
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


async def _runner(node: NodeSpec, _task: NodeTask) -> NodeYield:
    return NodeYield(status=YieldStatus.done, result_ref=f"result:{node.node_id}")


@pytest.mark.asyncio
async def test_api_lifecycle_trace_stream_and_cancel(pg: str) -> None:
    from harness.api.app import create_app
    from harness.durability.postgres.backend import PostgresBackend

    await _reset_and_migrate(pg)
    backend = PostgresBackend(pg, background_flush=False)
    app = create_app(
        backend=backend,
        spec=_spec(),
        node_runner=_runner,
        idgen=SequentialIdGen(1500),
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/healthz")
        created = await client.post("/v1/runs", json={"input": {"goal": "demo"}})
        run_id = created.json()["run_id"]
        loaded = await client.get(f"/v1/runs/{run_id}")
        trace = await client.get(f"/v1/runs/{run_id}/trace")
        stream = await client.get(f"/v1/runs/{run_id}/stream")
        cancelled = await client.post(f"/v1/runs/{run_id}/cancel")
        metrics = await client.get("/metrics")

    assert health.json() == {"ok": True}
    assert created.status_code == 200
    assert created.json()["status"] == "succeeded"
    assert loaded.json()["status"] == "succeeded"
    assert [event["kind"] for event in trace.json()["events"]] == [
        "node_started",
        "yield",
        "node_started",
        "yield",
        "run_finished",
    ]
    assert "event: node_transition" in stream.text
    assert stream.text.index("node_started") < stream.text.index("run_finished")
    assert cancelled.json()["status"] == "cancelled"
    assert "harness_runs_total" in metrics.text
