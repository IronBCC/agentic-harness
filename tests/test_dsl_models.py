from __future__ import annotations

import pytest
from harness.dsl.models import AgentSpec, dumps_yaml, loads_yaml
from pydantic import ValidationError


def _spec_dict() -> dict[str, object]:
    return {
        "spec_id": "support-agent",
        "version": 1,
        "description": "Resolve a support request using read tools and one keyed write.",
        "fact_types": [
            {
                "name": "ticket",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
                "scope_allowed": ["run", "tenant"],
                "promote_auto": 5,
                "promote_review": 2,
            }
        ],
        "tools": [
            {
                "tool_id": "crm.lookup",
                "side_effect": "read",
                "idempotency": "none",
                "freshness": "session",
                "auth_mode": "service",
                "emits_fact_types": ["ticket"],
            },
            {
                "tool_id": "crm.update",
                "side_effect": "write",
                "idempotency": "keyed",
                "freshness": "volatile",
                "auth_mode": "user_passthrough",
                "requires_approval": True,
            },
        ],
        "models": [
            {
                "name": "planner",
                "provider": "openai_compat",
                "model": "gemma4",
                "params": {"temperature": 0},
                "required_caps": ["guided_decoding"],
            }
        ],
        "nodes": [
            {
                "node_id": "plan",
                "kind": "planner",
                "model": "planner",
                "capability_set": ["crm.lookup"],
                "max_context_tokens": 4096,
                "decision_mode": "micro",
                "prompt_template": "support.plan.v1",
                "output_schema": "ticket",
                "retry": {"max_attempts": 2, "backoff_ms": 100, "jitter": 0.0},
                "cost_estimate": {"tokens": 800, "usd": 0.01, "wall_ms": 900, "llm_calls": 1},
                "cache": {"enabled": True},
                "speculative_prefetch": ["crm.lookup"],
                "can_elicit": True,
            },
            {
                "node_id": "write",
                "kind": "tool_chain",
                "model": "planner",
                "capability_set": ["crm.update"],
                "max_context_tokens": 2048,
                "prompt_template": "support.write.v1",
                "retry": {"max_attempts": 0, "backoff_ms": 0, "jitter": 0.0},
                "cost_estimate": {"tokens": 200, "usd": 0.002, "wall_ms": 500, "llm_calls": 1},
            },
        ],
        "edges": [{"from_node": "plan", "to_node": "write"}],
        "budget_policy": {
            "pools": {
                "interactive": {"tokens": 10000, "usd": 1.0, "wall_ms": 30000, "llm_calls": 10}
            },
            "lease_decay": 0.6,
            "pressure_thresholds": [0.5, 0.15],
            "degradation": {"on_yellow": "defer_optional", "on_red": "wrap_up"},
            "max_depth_circuit_breaker": 12,
            "elicitation": {
                "max_questions_per_run": 3,
                "batch_window_ms": 500,
                "timeout_s": 900,
                "on_timeout": "best_effort",
            },
        },
        "policies": {"tool_policy": "default", "redaction_policy": "default"},
        "evals": [
            {
                "case_id": "happy-path",
                "input": {"message": "help with my ticket"},
                "expected": {"status": "done"},
            }
        ],
    }


def test_agent_spec_yaml_round_trip_is_byte_stable_after_normalization() -> None:
    spec = AgentSpec.model_validate(_spec_dict())

    first = dumps_yaml(spec)
    second = dumps_yaml(loads_yaml(first))

    assert second == first
    assert loads_yaml(second) == spec


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unexpected",), True),
        (("nodes", 0, "unexpected"), True),
        (("tools", 0, "unexpected"), True),
        (("budget_policy", "elicitation", "unexpected"), True),
    ],
)
def test_agent_spec_rejects_unknown_fields(path: tuple[object, ...], value: object) -> None:
    raw = _spec_dict()
    cursor: object = raw
    for part in path[:-1]:
        if isinstance(part, int):
            cursor = cursor[part]  # type: ignore[index]
        else:
            cursor = cursor[part]  # type: ignore[index]
    assert isinstance(cursor, dict)
    cursor[path[-1]] = value

    with pytest.raises(ValidationError):
        AgentSpec.model_validate(raw)
