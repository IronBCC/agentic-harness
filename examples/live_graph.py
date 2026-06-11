from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Protocol
from uuid import UUID

from harness.dsl.models import AgentSpec, NodeSpec
from harness.durability.postgres.backend import PostgresBackend
from harness.durability.postgres.migrate import apply
from harness.engine.executor import Executor
from harness.engine.lifecycle import parse_yield
from harness.errors import SchemaViolation
from harness.models.adapters.openai_compat import OpenAICompatAdapter
from harness.types import (
    LLMEvent,
    LLMRequest,
    Message,
    ModelBinding,
    NodeTask,
    NodeTaskSnapshot,
    NodeYield,
    Principal,
    RunInit,
    RunState,
)
from harness.util import Uuid4IdGen

DEFAULT_BASE_URL = "https://ironbccllm.tail0cc1d4.ts.net:8002/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_MODEL = "gemma4"
DEFAULT_DSN = "postgresql://harness:harness@localhost:55432/harness"


class Adapter(Protocol):
    def complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]: ...


def build_spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "spec_id": "live-graph-demo",
            "version": 1,
            "description": "Two-node live Gemma graph demo",
            "fact_types": [{"name": "node_result", "schema": {"type": "object"}}],
            "tools": [],
            "models": [{"name": "ironbcc-gemma", "provider": "openai_compat", "model": "gemma"}],
            "nodes": [
                {
                    "node_id": "planner",
                    "kind": "planner",
                    "model": "ironbcc-gemma",
                    "max_context_tokens": 4096,
                    "prompt_template": "plan tool failure handling",
                    "retry": {"max_attempts": 0, "backoff_ms": 100, "jitter": 0.0},
                    "cost_estimate": {
                        "tokens": 300,
                        "usd": 0.0,
                        "wall_ms": 1000,
                        "llm_calls": 1,
                    },
                },
                {
                    "node_id": "synthesizer",
                    "kind": "synthesizer",
                    "model": "ironbcc-gemma",
                    "max_context_tokens": 4096,
                    "prompt_template": "summarize plan",
                    "retry": {"max_attempts": 0, "backoff_ms": 100, "jitter": 0.0},
                    "cost_estimate": {
                        "tokens": 300,
                        "usd": 0.0,
                        "wall_ms": 1000,
                        "llm_calls": 1,
                    },
                },
            ],
            "edges": [{"from_node": "planner", "to_node": "synthesizer"}],
            "budget_policy": {
                "pools": {
                    "interactive": {
                        "tokens": 2000,
                        "usd": 1.0,
                        "wall_ms": 30000,
                        "llm_calls": 4,
                    }
                },
                "degradation": {},
            },
            "policies": {},
            "evals": [{"case_id": "live-smoke", "input": {}, "expected": {}}],
        }
    )


def build_request(node: NodeSpec, task: NodeTask, model: str, goal: str) -> LLMRequest:
    return LLMRequest(
        binding=ModelBinding(
            name="ironbcc-gemma",
            provider="openai_compat",
            model=model,
        ),
        messages=[
            Message(
                role="system",
                content=(
                    "Return exactly one JSON object. No markdown. "
                    'The object must match: {"status":"done","facts":[...],"result_ref":"..."}'
                ),
            ),
            Message(
                role="user",
                content=(
                    f'Node "{node.node_id}" is running in a two-node agent graph. '
                    f"Goal: {goal}. "
                    f"Task input: {json.dumps(task.input, sort_keys=True)}. "
                    "Return a done NodeYield JSON object with one fact describing this "
                    "node's output "
                    f'and result_ref set to "artifact:{node.node_id}".'
                ),
            ),
        ],
    )


def clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned.removeprefix("json").strip()
    return cleaned


async def run_node(
    adapter: Adapter,
    model: str,
    goal: str,
    node: NodeSpec,
    task: NodeTask,
) -> NodeYield:
    text = ""
    async for event in adapter.complete(build_request(node, task, model, goal)):
        if event.type == "token":
            text += str(event.data.get("text", ""))
    if not text.strip():
        raise SchemaViolation(f"node {node.node_id} produced no token text")
    return parse_yield(clean_json_text(text))


async def run_graph(
    *,
    dsn: str,
    adapter: Adapter,
    model: str,
    run_id: UUID,
    goal: str,
) -> tuple[RunState, list[NodeTaskSnapshot]]:
    await apply(dsn)
    backend = PostgresBackend(dsn, background_flush=False)
    spec = build_spec()

    async def node_runner(node: NodeSpec, task: NodeTask) -> NodeYield:
        return await run_node(adapter, model, goal, node, task)

    executor = Executor(
        backend=backend,
        spec=spec,
        node_runner=node_runner,
        idgen=Uuid4IdGen(),
        worker="live-graph-example",
    )
    run = RunInit(
        run_id=run_id,
        root_run_id=run_id,
        tenant_id="example",
        principal=Principal(user_id="local"),
        spec_id=spec.spec_id,
        spec_version=spec.version,
        request_class="interactive",
        budget={},
    )
    await executor.seed_run(run, input={"goal": goal})
    await executor.run_until_idle()
    return await backend.load(run_id), await backend.list_tasks(run_id)


async def main() -> None:
    run_id = uuid.uuid4()
    loaded, tasks = await run_graph(
        dsn=os.environ.get("HARNESS_DSN", DEFAULT_DSN),
        adapter=OpenAICompatAdapter(
            base_url=os.environ.get("GEMMA_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ.get("GEMMA_API_KEY", DEFAULT_API_KEY),
        ),
        model=os.environ.get("GEMMA_MODEL", DEFAULT_MODEL),
        run_id=run_id,
        goal=os.environ.get("GRAPH_GOAL", "handle a failed restaurant reservation tool call"),
    )

    print("RUN_ID:", loaded.run_id)
    print("RUN_STATUS:", loaded.status)
    print("RUN_RESULT:", json.dumps(loaded.result, indent=2, sort_keys=True))
    print("TASKS:")
    for task in tasks:
        print(
            json.dumps(
                {
                    "node_id": task.node_id,
                    "state": task.state,
                    "attempt": task.attempt,
                    "input": task.input,
                },
                sort_keys=True,
            )
        )
    print("EVENTS:")
    for event in loaded.events:
        print(
            json.dumps(
                {
                    "seq": event.seq,
                    "node_id": event.node_id,
                    "kind": event.kind.value,
                    "payload": event.payload,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
