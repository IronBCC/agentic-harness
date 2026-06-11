from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from harness.dsl.models import NodeSpec
from harness.types import NodeTask, YieldStatus
from harness.util import FrozenClock


def _node(
    *,
    max_attempts: int = 3,
    backoff_ms: int = 100,
    jitter: float = 0.0,
    escalation: list[str] | None = None,
) -> NodeSpec:
    return NodeSpec.model_validate(
        {
            "node_id": "triage",
            "kind": "leaf",
            "model": "gemma4-nano",
            "escalation": escalation or [],
            "max_context_tokens": 1024,
            "prompt_template": "triage",
            "retry": {
                "max_attempts": max_attempts,
                "backoff_ms": backoff_ms,
                "jitter": jitter,
            },
            "cost_estimate": {
                "tokens": 1,
                "usd": 0.0,
                "wall_ms": 1,
                "llm_calls": 1,
            },
        }
    )


def _task(*, attempt: int = 0, input: dict[str, object] | None = None) -> NodeTask:
    return NodeTask(
        task_id=UUID(int=700),
        run_id=UUID(int=701),
        node_id="triage",
        attempt=attempt,
        input=input or {},
    )


def test_infra_retry_uses_exact_exponential_backoff_with_frozen_clock() -> None:
    from harness.engine.retry import plan_infra_retry

    clock = FrozenClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    first = plan_infra_retry(_node(backoff_ms=250), _task(attempt=0), clock)
    third = plan_infra_retry(_node(backoff_ms=250), _task(attempt=2), clock)

    assert first.kind == "reschedule"
    assert first.next_attempt == 1
    assert first.available_at == clock.now() + timedelta(milliseconds=250)
    assert third.kind == "reschedule"
    assert third.next_attempt == 3
    assert third.available_at == clock.now() + timedelta(milliseconds=1000)


def test_infra_retry_exhaustion_returns_error_yield_without_planner_fact() -> None:
    from harness.engine.retry import plan_infra_retry

    decision = plan_infra_retry(
        _node(max_attempts=2),
        _task(attempt=2),
        FrozenClock(datetime(2026, 1, 1, tzinfo=UTC)),
    )

    assert decision.kind == "terminal"
    assert decision.node_yield is not None
    assert decision.node_yield.status == YieldStatus.error
    assert decision.node_yield.error_class == "infra_retryable"
    assert decision.node_yield.facts == []


def test_schema_violation_retries_same_model_twice_then_walks_escalation_ladder() -> None:
    from harness.engine.retry import active_model, plan_schema_violation

    clock = FrozenClock(datetime(2026, 1, 1, tzinfo=UTC))
    node = _node(backoff_ms=10, escalation=["gemma4", "gemma4-pro"])

    first = plan_schema_violation(node, _task(input={}), clock)
    second = plan_schema_violation(node, _task(input=first.next_input), clock)
    escalated = plan_schema_violation(node, _task(input=second.next_input), clock)
    after_one_retry_on_escalated = plan_schema_violation(
        node,
        _task(input=escalated.next_input),
        clock,
    )
    after_two_retries_on_escalated = plan_schema_violation(
        node,
        _task(input=after_one_retry_on_escalated.next_input),
        clock,
    )
    second_escalation = plan_schema_violation(
        node,
        _task(input=after_two_retries_on_escalated.next_input),
        clock,
    )

    assert first.kind == "reschedule"
    assert active_model(node, first.next_input) == "gemma4-nano"
    assert first.next_input["schema_retries"] == 1
    assert active_model(node, second.next_input) == "gemma4-nano"
    assert second.next_input["schema_retries"] == 2
    assert escalated.kind == "reschedule"
    assert active_model(node, escalated.next_input) == "gemma4"
    assert escalated.next_input["model_index"] == 1
    assert escalated.next_input["schema_retries"] == 0
    assert active_model(node, after_two_retries_on_escalated.next_input) == "gemma4"
    assert active_model(node, second_escalation.next_input) == "gemma4-pro"


def test_schema_violation_exhaustion_returns_error_yield() -> None:
    from harness.engine.retry import plan_schema_violation

    node = _node(escalation=["gemma4"])
    decision = plan_schema_violation(
        node,
        _task(input={"model_index": 1, "schema_retries": 2}),
        FrozenClock(datetime(2026, 1, 1, tzinfo=UTC)),
    )

    assert decision.kind == "terminal"
    assert decision.node_yield is not None
    assert decision.node_yield.status == YieldStatus.error
    assert decision.node_yield.error_class == "schema_violation"


def test_attempt_invariant_idempotency_key_excludes_attempt() -> None:
    from harness.engine.retry import attempt_invariant_idempotency_key

    run_id = UUID(int=800)

    assert attempt_invariant_idempotency_key(run_id, "triage", 3, attempt=0) == (
        attempt_invariant_idempotency_key(run_id, "triage", 3, attempt=4)
    )
    assert attempt_invariant_idempotency_key(run_id, "triage", 3, attempt=0) != (
        attempt_invariant_idempotency_key(run_id, "triage", 4, attempt=0)
    )
