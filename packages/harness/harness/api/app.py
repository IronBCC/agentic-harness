"""FastAPI app for run lifecycle and trace streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from harness.dsl.models import AgentSpec
from harness.durability.postgres.backend import PostgresBackend
from harness.engine.executor import Executor, NodeRunner
from harness.types import Event, Principal, RunInit
from harness.util import IdGen


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: dict[str, object] = Field(default_factory=dict)
    tenant_id: str = "tenant-a"
    user_id: str = "user-a"


def create_app(
    *,
    backend: PostgresBackend,
    spec: AgentSpec,
    node_runner: NodeRunner,
    idgen: IdGen,
) -> FastAPI:
    app = FastAPI(title="Agentic Harness")

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return "harness_runs_total 0\n"

    @app.post("/v1/runs")
    async def create_run(body: RunCreate) -> dict[str, object]:
        run_id = idgen.new()
        executor = Executor(
            backend=backend,
            spec=spec,
            node_runner=node_runner,
            idgen=idgen,
            worker="api-worker",
        )
        await executor.seed_run(
            RunInit(
                run_id=run_id,
                root_run_id=run_id,
                tenant_id=body.tenant_id,
                principal=Principal(user_id=body.user_id),
                spec_id=spec.spec_id,
                spec_version=spec.version,
                request_class="interactive",
                budget={},
            ),
            input=body.input,
        )
        await executor.run_until_idle()
        loaded = await backend.load(run_id)
        return {"run_id": str(run_id), "status": loaded.status}

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, object]:
        loaded = await backend.load(_uuid(run_id))
        return loaded.model_dump(mode="json", exclude={"events"})

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict[str, object]:
        await backend.update_run(_uuid(run_id), status="cancelled")
        return {"run_id": run_id, "status": "cancelled"}

    @app.get("/v1/runs/{run_id}/trace")
    async def trace(run_id: str) -> dict[str, object]:
        loaded = await backend.load(_uuid(run_id))
        return {"events": [_event_payload(event) for event in loaded.events]}

    @app.get("/v1/runs/{run_id}/stream")
    async def stream(run_id: str) -> StreamingResponse:
        async def generate() -> AsyncIterator[str]:
            loaded = await backend.load(_uuid(run_id))
            for event in loaded.events:
                event_type = (
                    "run_finished" if event.kind.value == "run_finished" else "node_transition"
                )
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(_event_payload(event), sort_keys=True)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


def _uuid(value: str) -> UUID:
    return UUID(value)


def _event_payload(event: Event) -> dict[str, object]:
    return {
        "seq": event.seq,
        "node_id": event.node_id,
        "kind": event.kind.value,
        "payload": event.payload,
    }
