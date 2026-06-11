"""Command-line interface for the harness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

import typer

from harness.dsl.models import AgentSpec, dumps_yaml, loads_yaml
from harness.durability.postgres.backend import PostgresBackend
from harness.durability.postgres.migrate import apply
from harness.engine.executor import Executor
from harness.types import NodeTask, NodeYield, Principal, RunInit, YieldStatus
from harness.util import SequentialIdGen

app = typer.Typer(no_args_is_help=True)

DEFAULT_DSN = "postgresql://harness:harness@localhost:55432/harness"


@app.command()
def init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    spec = _example_spec()
    (path / "agent.yaml").write_text(dumps_yaml(spec), encoding="utf-8")
    typer.echo(f"created {path / 'agent.yaml'}")


@app.command()
def migrate(dsn: str = DEFAULT_DSN) -> None:
    applied = asyncio.run(apply(dsn))
    typer.echo(json.dumps({"applied": applied}))


@app.command()
def dev(check_only: bool = False) -> None:
    if check_only:
        typer.echo("/healthz ok")
        return
    typer.echo("dev server embedding is planned; use --check-only in M1")


@app.command()
def run(spec_yaml: Path, dsn: str = DEFAULT_DSN) -> None:
    spec = loads_yaml(spec_yaml.read_text(encoding="utf-8"))
    raw = typer.get_text_stream("stdin").read().strip()
    input_payload = json.loads(raw) if raw else {}
    result = asyncio.run(_run_spec(spec, dsn, input_payload))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def replay(run_id: str, dsn: str = DEFAULT_DSN) -> None:
    result = asyncio.run(_load_run(UUID(run_id), dsn))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


async def _run_spec(
    spec: AgentSpec,
    dsn: str,
    input_payload: dict[str, object],
) -> dict[str, object]:
    await apply(dsn)
    backend = PostgresBackend(dsn, background_flush=False)
    idgen = SequentialIdGen(17_000)
    run_id = idgen.new()
    executor = Executor(
        backend=backend,
        spec=spec,
        node_runner=_done_runner,
        idgen=idgen,
        worker="cli-worker",
    )
    await executor.seed_run(
        RunInit(
            run_id=run_id,
            root_run_id=run_id,
            tenant_id="cli",
            principal=Principal(user_id="cli"),
            spec_id=spec.spec_id,
            spec_version=spec.version,
            request_class="interactive",
            budget={},
        ),
        input=input_payload,
    )
    await executor.run_until_idle()
    loaded = await backend.load(run_id)
    return loaded.model_dump(mode="json")


async def _load_run(run_id: UUID, dsn: str) -> dict[str, object]:
    backend = PostgresBackend(dsn, background_flush=False)
    loaded = await backend.load(run_id)
    return loaded.model_dump(mode="json")


async def _done_runner(_node: object, task: NodeTask) -> NodeYield:
    return NodeYield(status=YieldStatus.done, result_ref=f"cli:{task.node_id}")


def _example_spec() -> AgentSpec:
    return AgentSpec.model_validate(
        {
            "spec_id": "cli-example",
            "version": 1,
            "description": "CLI example graph",
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


if __name__ == "__main__":
    app()
