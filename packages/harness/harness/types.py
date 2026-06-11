"""Canonical kernel types shared across modules."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

RunId = UUID
NodeId = str
Tenant = str


class EventKind(StrEnum):
    node_started = "node_started"
    prompt_assembled = "prompt_assembled"
    llm_call = "llm_call"
    tool_call = "tool_call"
    fact_emitted = "fact_emitted"
    yielded = "yield"
    spawn_proposed = "spawn_proposed"
    spawn_approved = "spawn_approved"
    spawn_denied = "spawn_denied"
    checkpoint = "checkpoint"
    run_finished = "run_finished"


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int
    node_id: NodeId
    kind: EventKind
    payload: dict[str, object] = Field(default_factory=dict)
    idempotency_key: str
    barrier: bool = False


class YieldStatus(StrEnum):
    done = "done"
    need_capability = "need_capability"
    low_confidence = "low_confidence"
    need_user_input = "need_user_input"
    wrap_up_ack = "wrap_up_ack"
    error = "error"


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    kind: Literal["select", "text"]
    options: list[str] = Field(default_factory=list)
    maps_to_fact_type: str | None = None


class NodeYield(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: YieldStatus
    facts: list[dict[str, object]] = Field(default_factory=list)
    result_ref: str | None = None
    confidence: float | None = None
    need: str | None = None
    tried: list[str] = Field(default_factory=list)
    best_effort: dict[str, object] | None = None
    reason: str | None = None
    questions: list[Question] = Field(default_factory=list)
    summary_fact: dict[str, object] | None = None
    error_class: str | None = Field(default=None, alias="class")
    detail: dict[str, object] | None = None


class NodeTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    run_id: RunId
    node_id: NodeId
    attempt: int
    input: dict[str, object] = Field(default_factory=dict)


class Principal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None
    claims: dict[str, object] = Field(default_factory=dict)


class CallCtx(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: Tenant
    principal: Principal
    run_id: RunId
    node_id: NodeId
    root_run_id: RunId


class CassetteMode(StrEnum):
    record = "record"
    replay = "replay"
    hybrid = "hybrid"
    live = "live"


class ModelBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    model: str
    params: dict[str, object] = Field(default_factory=dict)
    required_caps: list[str] = Field(default_factory=list)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    binding: ModelBinding
    messages: list[Message]
    output_schema: dict[str, object] | None = None
    stream: bool = True
    cassette_mode: CassetteMode = CassetteMode.live


class LLMEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["token", "tool_call", "usage", "done"]
    data: dict[str, object] = Field(default_factory=dict)

