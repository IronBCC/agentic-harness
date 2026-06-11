from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from uuid import UUID

import asyncpg
import pytest
from harness.types import LLMEvent, LLMRequest


class FakeAdapter:
    def complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        return self._complete(req)

    async def _complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        node_id = "synthesizer" if "synthesizer" in req.messages[-1].content else "planner"
        yield LLMEvent(
            type="token",
            data={
                "text": (
                    '{"status":"done","facts":[{"node":"'
                    f'{node_id}'
                    '"}],"result_ref":"artifact:'
                    f'{node_id}'
                    '"}'
                )
            },
        )
        yield LLMEvent(type="done")


async def _reset_and_migrate(dsn: str) -> None:
    from harness.durability.postgres.migrate import apply

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()
    await apply(dsn)


def _load_example() -> ModuleType:
    path = Path("examples/live_graph.py")
    spec = importlib.util.spec_from_file_location("live_graph_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_graph_example_spec_is_two_node_graph() -> None:
    module = _load_example()

    spec = module.build_spec()

    assert [node.node_id for node in spec.nodes] == ["planner", "synthesizer"]
    assert [(edge.from_node, edge.to_node) for edge in spec.edges] == [("planner", "synthesizer")]


@pytest.mark.asyncio
async def test_live_graph_example_runs_graph_with_fake_adapter(pg: str) -> None:
    module = _load_example()
    run_id = UUID(int=1200)
    await _reset_and_migrate(pg)

    loaded, tasks = await module.run_graph(
        dsn=pg,
        adapter=FakeAdapter(),
        model="fake-gemma",
        run_id=run_id,
        goal="handle a failed reservation",
    )

    assert loaded.status == "succeeded"
    assert loaded.result == {
        "terminal_node": "synthesizer",
        "result_ref": "artifact:synthesizer",
    }
    assert [(task.node_id, task.state) for task in tasks] == [
        ("planner", "done"),
        ("synthesizer", "done"),
    ]
    assert [event.kind.value for event in loaded.events] == [
        "node_started",
        "yield",
        "node_started",
        "yield",
        "run_finished",
    ]
