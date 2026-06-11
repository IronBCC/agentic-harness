from __future__ import annotations

from uuid import UUID

from harness.dsl.models import AgentSpec, NodeSpec
from harness.durability.postgres.backend import PostgresBackend
from harness.engine.executor import Executor
from harness.types import NodeTask, NodeYield, Principal, RunInit, RunState, YieldStatus
from harness.util import Uuid4IdGen

FACTS = {
    "planner": {"node": "planner", "summary": "triage support request"},
    "kb_leaf": {"node": "kb_leaf", "summary": "found refund policy"},
    "ticket_leaf": {"node": "ticket_leaf", "summary": "updated support ticket"},
    "synthesizer": {"node": "synthesizer", "summary": "customer can receive refund"},
}


def build_spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "spec_id": "scenario-a",
            "version": 1,
            "description": "Support-ish replay scenario",
            "fact_types": [{"name": "result", "schema": {"type": "object"}}],
            "tools": [],
            "models": [{"name": "fake", "provider": "fake", "model": "fake"}],
            "nodes": [
                _node("planner"),
                _node("kb_leaf"),
                _node("ticket_leaf"),
                _node("synthesizer"),
            ],
            "edges": [
                {"from_node": "planner", "to_node": "kb_leaf"},
                {"from_node": "planner", "to_node": "ticket_leaf"},
                {"from_node": "kb_leaf", "to_node": "synthesizer"},
                {"from_node": "ticket_leaf", "to_node": "synthesizer"},
            ],
            "budget_policy": {
                "pools": {
                    "interactive": {
                        "tokens": 1000,
                        "usd": 1.0,
                        "wall_ms": 10000,
                        "llm_calls": 10,
                    }
                },
                "degradation": {},
            },
            "policies": {},
            "evals": [{"case_id": "scenario-a", "input": {}, "expected": {}}],
        }
    )


def build_executor(
    dsn: str,
    *,
    run_id: UUID,
) -> tuple[Executor, PostgresBackend, RunInit]:
    backend = PostgresBackend(dsn, background_flush=False)
    spec = build_spec()
    executor = Executor(
        backend=backend,
        spec=spec,
        node_runner=_runner,
        idgen=Uuid4IdGen(),
        worker="scenario-a",
        batch_size=2,
    )
    run = RunInit(
        run_id=run_id,
        root_run_id=run_id,
        tenant_id="scenario",
        principal=Principal(user_id="scenario"),
        spec_id=spec.spec_id,
        spec_version=spec.version,
        request_class="interactive",
        budget={},
    )
    return executor, backend, run


async def run_replay(dsn: str, *, run_id: UUID) -> tuple[RunState, list[dict[str, object]]]:
    executor, backend, run = build_executor(dsn, run_id=run_id)
    await executor.seed_run(run, input={"case": "scenario-a"})
    await executor.run_until_idle()
    loaded = await backend.load(run_id)
    return loaded, load_final_facts(loaded)


def load_final_facts(loaded: RunState) -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    for event in loaded.events:
        if event.kind.value == "yield":
            raw_facts = event.payload.get("facts", [])
            if isinstance(raw_facts, list):
                facts.extend(fact for fact in raw_facts if isinstance(fact, dict))
    return facts


async def _runner(node: NodeSpec, _task: NodeTask) -> NodeYield:
    return NodeYield(
        status=YieldStatus.done,
        facts=[FACTS[node.node_id]],
        result_ref=f"scenario-a:{node.node_id}",
    )


def _node(node_id: str) -> dict[str, object]:
    return {
        "node_id": node_id,
        "kind": "planner" if node_id == "planner" else "leaf",
        "model": "fake",
        "max_context_tokens": 1024,
        "prompt_template": node_id,
        "retry": {"max_attempts": 0, "backoff_ms": 0, "jitter": 0.0},
        "cost_estimate": {
            "tokens": 1,
            "usd": 0.0,
            "wall_ms": 1,
            "llm_calls": 0,
        },
    }
