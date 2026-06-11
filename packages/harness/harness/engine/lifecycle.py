"""Node lifecycle state machine and yield routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import ValidationError

from harness.errors import SchemaViolation
from harness.types import Event, EventKind, NodeYield, YieldStatus

SideEffect = Literal["pure", "read", "write"]


class NodePhase(StrEnum):
    pending = "pending"
    claimed = "claimed"
    assembling = "assembling"
    executing = "executing"
    yielded = "yielded"
    retry_wait = "retry_wait"
    completed = "completed"
    escalated = "escalated"
    waiting = "waiting"
    failed = "failed"


@dataclass
class NodeLifecycle:
    """Small state machine for node execution phases."""

    state: NodePhase = NodePhase.pending

    def claim(self) -> NodeLifecycle:
        self._move(NodePhase.pending, NodePhase.claimed)
        return self

    def lease_expired(self) -> NodeLifecycle:
        self._move(NodePhase.claimed, NodePhase.pending)
        return self

    def start_assembling(self) -> NodeLifecycle:
        self._move(NodePhase.claimed, NodePhase.assembling)
        return self

    def assembly_failed(self) -> NodeLifecycle:
        self._move(NodePhase.assembling, NodePhase.failed)
        return self

    def start_executing(self) -> NodeLifecycle:
        self._move(NodePhase.assembling, NodePhase.executing)
        return self

    def infra_retryable(self) -> NodeLifecycle:
        self._move(NodePhase.executing, NodePhase.retry_wait)
        return self

    def mark_yielded(self) -> NodeLifecycle:
        self._move(NodePhase.executing, NodePhase.yielded)
        return self

    def complete(self) -> NodeLifecycle:
        self._move(NodePhase.yielded, NodePhase.completed)
        return self

    def escalate(self) -> NodeLifecycle:
        self._move(NodePhase.yielded, NodePhase.escalated)
        return self

    def wait(self) -> NodeLifecycle:
        self._move(NodePhase.yielded, NodePhase.waiting)
        return self

    def fail(self) -> NodeLifecycle:
        self._move(NodePhase.yielded, NodePhase.failed)
        return self

    def _move(self, expected: NodePhase, next_state: NodePhase) -> None:
        if self.state != expected:
            raise SchemaViolation(f"invalid lifecycle transition: {self.state} -> {next_state}")
        self.state = next_state

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NodePhase):
            return self.state == other
        return super().__eq__(other)


@dataclass(frozen=True)
class YieldRoute:
    action: Literal["complete", "retry", "emit_fact"]
    terminal_state: Literal["completed", "retry_wait", "failed"]
    planner_visible: bool = False


def parse_yield(raw: str | dict[str, object]) -> NodeYield:
    """Strictly parse a node yield from JSON text or a mapping."""
    try:
        if isinstance(raw, str):
            parsed = json.loads(raw)
        else:
            parsed = raw
        if not isinstance(parsed, dict):
            raise TypeError("yield must be a JSON object")
        return NodeYield.model_validate(parsed)
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise SchemaViolation("node yield failed schema validation") from exc


def route_yield(raw: str | dict[str, object] | NodeYield) -> YieldRoute:
    """Map a parsed node yield to the lifecycle action owned by this milestone."""
    node_yield = raw if isinstance(raw, NodeYield) else parse_yield(raw)

    if node_yield.status == YieldStatus.done:
        return YieldRoute(action="complete", terminal_state="completed")

    if node_yield.status == YieldStatus.error:
        if node_yield.error_class in {"infra_retryable", "schema_violation"}:
            return YieldRoute(action="retry", terminal_state="retry_wait")
        if node_yield.error_class in {"tool_rejected", "semantic_failure", "policy_denied"}:
            return YieldRoute(action="emit_fact", terminal_state="failed", planner_visible=True)
        raise SchemaViolation("unknown node yield error class")

    if node_yield.status in {YieldStatus.need_capability, YieldStatus.low_confidence}:
        raise NotImplementedError("router/escalation routing lands in M2-09")

    if node_yield.status in {YieldStatus.need_user_input, YieldStatus.wrap_up_ack}:
        raise NotImplementedError("elicitation/wrap-up routing lands in M3-11")

    raise SchemaViolation("unsupported node yield status")


def tool_call_events(
    *,
    seq: int,
    node_id: str,
    tool_id: str,
    side_effect: SideEffect,
    payload: dict[str, object],
    idempotency_key: str,
) -> list[Event]:
    """Return lifecycle events for a tool call, including write barriers."""
    tool_event = Event(
        seq=seq,
        node_id=node_id,
        kind=EventKind.tool_call,
        payload={"tool_id": tool_id, **payload},
        idempotency_key=idempotency_key,
    )
    if side_effect != "write":
        return [tool_event]

    checkpoint = Event(
        seq=seq,
        node_id=node_id,
        kind=EventKind.checkpoint,
        payload={"before": "tool_call", "tool_id": tool_id},
        idempotency_key=f"{idempotency_key}:checkpoint",
        barrier=True,
    )
    return [checkpoint, tool_event.model_copy(update={"seq": seq + 1})]
