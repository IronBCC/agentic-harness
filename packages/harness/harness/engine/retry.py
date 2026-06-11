"""Retry and model-escalation planning for node execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from harness.dsl.models import NodeSpec
from harness.types import NodeTask, NodeYield, YieldStatus
from harness.util import Clock

SAME_MODEL_SCHEMA_RETRIES = 2


@dataclass(frozen=True)
class RetryDecision:
    kind: Literal["reschedule", "terminal"]
    next_attempt: int | None = None
    available_at: datetime | None = None
    next_input: dict[str, object] = field(default_factory=dict)
    node_yield: NodeYield | None = None


def plan_infra_retry(node: NodeSpec, task: NodeTask, clock: Clock) -> RetryDecision:
    """Plan an infrastructure retry or terminal infra error yield."""
    if task.attempt >= node.retry.max_attempts:
        return RetryDecision(
            kind="terminal",
            node_yield=NodeYield.model_validate(
                {
                    "status": YieldStatus.error,
                    "class": "infra_retryable",
                    "detail": {
                        "attempt": task.attempt,
                        "max_attempts": node.retry.max_attempts,
                    },
                }
            ),
        )

    next_attempt = task.attempt + 1
    return RetryDecision(
        kind="reschedule",
        next_attempt=next_attempt,
        available_at=clock.now() + _backoff_delay(node, task.attempt),
        next_input=dict(task.input),
    )


def plan_schema_violation(node: NodeSpec, task: NodeTask, clock: Clock) -> RetryDecision:
    """Plan structured-output retry, model escalation, or terminal schema error."""
    model_index = _model_index(task.input)
    schema_retries = _schema_retries(task.input)

    if schema_retries < SAME_MODEL_SCHEMA_RETRIES:
        next_input = dict(task.input)
        next_input["model_index"] = model_index
        next_input["schema_retries"] = schema_retries + 1
        return RetryDecision(
            kind="reschedule",
            next_attempt=task.attempt + 1,
            available_at=clock.now() + _backoff_delay(node, task.attempt),
            next_input=next_input,
        )

    next_model_index = model_index + 1
    if next_model_index < len(_model_ladder(node)):
        next_input = dict(task.input)
        next_input["model_index"] = next_model_index
        next_input["schema_retries"] = 0
        return RetryDecision(
            kind="reschedule",
            next_attempt=task.attempt + 1,
            available_at=clock.now() + _backoff_delay(node, task.attempt),
            next_input=next_input,
        )

    return RetryDecision(
        kind="terminal",
        node_yield=NodeYield.model_validate(
            {
                "status": YieldStatus.error,
                "class": "schema_violation",
                "detail": {
                    "model": active_model(node, task.input),
                    "schema_retries": schema_retries,
                },
            }
        ),
    )


def active_model(node: NodeSpec, task_input: dict[str, object]) -> str:
    """Return the model selected by the retry/escalation state in task input."""
    ladder = _model_ladder(node)
    index = _model_index(task_input)
    if index < 0 or index >= len(ladder):
        return ladder[-1]
    return ladder[index]


def attempt_invariant_idempotency_key(
    run_id: UUID,
    node_id: str,
    call_seq: int,
    *,
    attempt: int,
) -> str:
    """Create a stable tool-call idempotency key that intentionally ignores attempt."""
    _ = attempt
    payload = f"{run_id}:{node_id}:{call_seq}".encode()
    return hashlib.sha256(payload).hexdigest()


def _backoff_delay(node: NodeSpec, attempt: int) -> timedelta:
    base_ms = node.retry.backoff_ms * (2**attempt)
    return timedelta(milliseconds=base_ms)


def _model_ladder(node: NodeSpec) -> list[str]:
    return [node.model, *node.escalation]


def _model_index(task_input: dict[str, object]) -> int:
    raw = task_input.get("model_index", 0)
    return raw if isinstance(raw, int) else 0


def _schema_retries(task_input: dict[str, object]) -> int:
    raw = task_input.get("schema_retries", 0)
    return raw if isinstance(raw, int) else 0
