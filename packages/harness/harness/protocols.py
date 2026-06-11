"""Public protocol boundaries for swappable harness components."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol
from uuid import UUID

from harness.types import CallCtx, Event, LLMEvent, LLMRequest, NodeTask


class RunInit(Protocol):
    """Run creation payload marker."""


class RunState(Protocol):
    """Replayed run state marker."""


class TaskTerminal(Protocol):
    """Task terminal payload marker."""


class FactIn(Protocol):
    """Fact creation payload marker."""


class Fact(Protocol):
    """Stored fact marker."""


class FactQuery(Protocol):
    """Fact query marker."""


class ArtifactMeta(Protocol):
    """Artifact metadata marker."""


class Artifact(Protocol):
    """Stored artifact marker."""


class CompactionPolicy(Protocol):
    """Compaction policy marker."""


class CompactionReport(Protocol):
    """Compaction result marker."""


class IngestReport(Protocol):
    """Tool ingestion result marker."""


class IndexCard(Protocol):
    """Tool search index card marker."""


class ToolSchema(Protocol):
    """Resolved tool schema marker."""


class ToolCall(Protocol):
    """Tool invocation payload marker."""


class ToolResult(Protocol):
    """Tool invocation result marker."""


class ModelCaps(Protocol):
    """Model capability marker."""


class ProviderManifest(Protocol):
    """Tool provider manifest marker."""


class RunCtx(Protocol):
    """Provider run context marker."""


class ProviderSession(Protocol):
    """Provider session marker."""


class Decision(Protocol):
    """Policy decision marker."""


class RedactionCfg(Protocol):
    """Redaction configuration marker."""


class DurabilityBackend(Protocol):
    async def create_run(self, run: RunInit) -> None: ...
    async def append(self, run_id: UUID, events: list[Event]) -> None: ...
    async def load(self, run_id: UUID) -> RunState: ...
    async def claim(self, worker: str, n: int) -> list[NodeTask]: ...
    async def heartbeat(self, worker: str, task_ids: list[UUID]) -> None: ...
    async def complete(self, task_id: UUID, terminal: TaskTerminal) -> None: ...
    async def reschedule(self, task_id: UUID, at: datetime, attempt: int) -> None: ...


class Transport(Protocol):
    async def notify(self, channel: str, payload: bytes) -> None: ...
    async def subscribe(self, channel: str) -> AsyncIterator[bytes]: ...


class MemoryPlane(Protocol):
    async def put_fact(self, fact: FactIn) -> Fact: ...
    async def query(self, q: FactQuery) -> list[Fact]: ...
    async def put_artifact(self, blob: bytes, meta: ArtifactMeta) -> Artifact: ...
    async def get_artifact(self, artifact_id: UUID) -> bytes: ...
    async def compact(self, root_run_id: UUID, policy: CompactionPolicy) -> CompactionReport: ...


class ToolRegistry(Protocol):
    async def ingest_mcp(self, endpoint: str, tenant: str) -> IngestReport: ...
    async def ingest_openapi(self, spec_url: str, tenant: str) -> IngestReport: ...
    async def search(self, query: str, tenant: str, k: int = 5) -> list[IndexCard]: ...
    async def resolve(self, tool_id: str, tenant: str) -> ToolSchema: ...
    async def invoke(self, call: ToolCall, ctx: CallCtx) -> ToolResult: ...


class ModelGateway(Protocol):
    def capabilities(self, binding: object) -> ModelCaps: ...
    async def complete(self, req: LLMRequest) -> AsyncIterator[LLMEvent]: ...


class ToolProvider(Protocol):
    def manifest(self) -> ProviderManifest: ...
    async def open_session(self, ctx: RunCtx) -> ProviderSession | None: ...
    async def invoke(
        self,
        call: ToolCall,
        ctx: CallCtx,
        session: ProviderSession | None,
    ) -> ToolResult: ...


class PolicyEngine(Protocol):
    async def check(self, call: ToolCall, ctx: CallCtx) -> Decision: ...
    def redact(self, fact: FactIn, cfg: RedactionCfg) -> FactIn: ...

