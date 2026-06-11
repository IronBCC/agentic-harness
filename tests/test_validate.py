from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from harness.dsl.models import AgentSpec
from harness.dsl.validate import validate

RegistryView = dict[str, object]
CapsView = dict[str, dict[str, object]]
Mutator = Callable[[dict[str, Any]], None]


def _spec(overrides: Mutator | None = None) -> AgentSpec:
    raw: dict[str, Any] = {
        "spec_id": "validator-demo",
        "version": 1,
        "description": "Validator demo spec",
        "fact_types": [
            {
                "name": "ticket",
                "schema": {"type": "object"},
                "scope_allowed": ["run", "tenant"],
                "promote_auto": 5,
                "promote_review": 2,
            },
            {
                "name": "summary",
                "schema": {"type": "object"},
                "scope_allowed": ["run"],
                "promote_auto": 4,
                "promote_review": 1,
            },
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
                "required_caps": ["guided_decoding"],
            },
            {
                "name": "fallback",
                "provider": "openai_compat",
                "model": "gemma4-nano",
                "required_caps": [],
            },
        ],
        "nodes": [
            {
                "node_id": "plan",
                "kind": "planner",
                "model": "planner",
                "escalation": ["fallback"],
                "capability_set": ["crm.lookup"],
                "max_context_tokens": 4096,
                "prompt_template": "demo.plan",
                "output_schema": "ticket",
                "retry": {"max_attempts": 2, "backoff_ms": 100, "jitter": 0.0},
                "cost_estimate": {"tokens": 1000, "usd": 0.01, "wall_ms": 1000, "llm_calls": 1},
                "can_elicit": True,
            },
            {
                "node_id": "write",
                "kind": "tool_chain",
                "model": "planner",
                "capability_set": ["crm.update"],
                "max_context_tokens": 2048,
                "prompt_template": "demo.write",
                "retry": {"max_attempts": 0, "backoff_ms": 0, "jitter": 0.0},
                "cost_estimate": {"tokens": 500, "usd": 0.01, "wall_ms": 500, "llm_calls": 1},
            },
        ],
        "edges": [{"from_node": "plan", "to_node": "write"}],
        "budget_policy": {
            "pools": {
                "interactive": {"tokens": 5000, "usd": 0.5, "wall_ms": 10000, "llm_calls": 5},
                "background": {"tokens": 10000, "usd": 1.0, "wall_ms": 60000, "llm_calls": 10},
            },
            "degradation": {},
        },
        "policies": {},
        "evals": [{"case_id": "happy", "input": {}, "expected": {}}],
    }
    if overrides is not None:
        overrides(raw)
    return AgentSpec.model_validate(raw)


def _registry(*extra: str, missing: set[str] | None = None) -> RegistryView:
    tool_ids = {"crm.lookup", "crm.update", *extra} - (missing or set())
    return {"tools": sorted(tool_ids)}


def _caps(
    *,
    planner_caps: set[str] | None = None,
    planner_context: int = 8192,
    fallback_context: int = 4096,
) -> CapsView:
    caps = {"guided_decoding"} if planner_caps is None else planner_caps
    return {
        "planner": {
            "caps": sorted(caps),
            "max_context": planner_context,
        },
        "fallback": {"caps": [], "max_context": fallback_context},
    }


def _codes(
    spec: AgentSpec,
    registry: RegistryView | None = None,
    caps: CapsView | None = None,
) -> set[str]:
    return {
        violation.code
        for violation in validate(spec, registry or _registry(), caps or _caps())
    }


def _assert_rule(good_a: AgentSpec, good_b: AgentSpec, bad: AgentSpec, code: str) -> None:
    assert code not in _codes(good_a)
    assert code not in _codes(good_b)
    assert code in _codes(bad)


def test_v001_unreachable_node() -> None:
    def add_reachable(raw: dict[str, Any]) -> None:
        raw["nodes"].append(deepcopy(raw["nodes"][1]) | {"node_id": "audit"})
        raw["edges"].append({"from_node": "write", "to_node": "audit"})

    def add_unreachable(raw: dict[str, Any]) -> None:
        raw["nodes"].append(deepcopy(raw["nodes"][1]) | {"node_id": "orphan"})

    _assert_rule(_spec(), _spec(add_reachable), _spec(add_unreachable), "V001")


def test_v002_unresolved_tool_ref() -> None:
    def add_bound_tool(raw: dict[str, Any]) -> None:
        raw["tools"].append(
            {
                "tool_id": "crm.audit",
                "side_effect": "read",
                "idempotency": "none",
                "freshness": "session",
                "auth_mode": "service",
            }
        )
        raw["nodes"][0]["capability_set"].append("crm.audit")

    def add_unbound_tool(raw: dict[str, Any]) -> None:
        raw["nodes"][0]["capability_set"].append("crm.missing")

    assert "V002" not in _codes(_spec(), _registry(), _caps())
    assert "V002" not in _codes(_spec(add_bound_tool), _registry("crm.audit"), _caps())
    assert "V002" in _codes(_spec(add_unbound_tool), _registry(), _caps())
    assert "V002" in _codes(_spec(), _registry(missing={"crm.lookup"}), _caps())


def test_v003_unresolved_model_ref() -> None:
    def use_fallback(raw: dict[str, Any]) -> None:
        raw["nodes"][1]["model"] = "fallback"

    def use_missing_model(raw: dict[str, Any]) -> None:
        raw["nodes"][0]["model"] = "not-declared"

    _assert_rule(_spec(), _spec(use_fallback), _spec(use_missing_model), "V003")


def test_v004_model_caps_and_context() -> None:
    good = _spec()
    good_caps_without_requirement = _spec(
        lambda raw: raw["models"][0].update({"required_caps": []})
    )
    bad_missing_cap = _spec()
    bad_small_context = _spec()

    assert "V004" not in _codes(good, _registry(), _caps())
    assert "V004" not in _codes(
        good_caps_without_requirement,
        _registry(),
        _caps(planner_caps=set()),
    )
    assert "V004" in _codes(
        bad_missing_cap,
        _registry(),
        _caps(planner_caps=set()),
    )
    assert "V004" in _codes(
        bad_small_context,
        _registry(),
        _caps(planner_context=1024),
    )


def test_v005_write_none_requires_approval_and_zero_retry() -> None:
    def write_none_safe(raw: dict[str, Any]) -> None:
        raw["tools"][1]["idempotency"] = "none"
        raw["tools"][1]["requires_approval"] = True
        raw["nodes"][1]["retry"]["max_attempts"] = 0

    def write_none_unsafe(raw: dict[str, Any]) -> None:
        raw["tools"][1]["idempotency"] = "none"
        raw["tools"][1]["requires_approval"] = False

    def write_none_retrying(raw: dict[str, Any]) -> None:
        raw["tools"][1]["idempotency"] = "none"
        raw["nodes"][1]["retry"]["max_attempts"] = 1

    assert "V005" not in _codes(_spec(), _registry(), _caps())
    assert "V005" not in _codes(_spec(write_none_safe), _registry(), _caps())
    assert "V005" in _codes(_spec(write_none_unsafe), _registry(), _caps())
    assert "V005" in _codes(_spec(write_none_retrying), _registry(), _caps())


def test_v006_undefined_fact_type_reference() -> None:
    def emit_summary(raw: dict[str, Any]) -> None:
        raw["tools"][0]["emits_fact_types"].append("summary")

    def emit_missing(raw: dict[str, Any]) -> None:
        raw["tools"][0]["emits_fact_types"].append("unknown_fact")

    _assert_rule(_spec(), _spec(emit_summary), _spec(emit_missing), "V006")


def test_v007_empty_evals() -> None:
    def add_eval(raw: dict[str, Any]) -> None:
        raw["evals"].append({"case_id": "second", "input": {}, "expected": {}})

    def empty_evals(raw: dict[str, Any]) -> None:
        raw["evals"] = []

    _assert_rule(_spec(), _spec(add_eval), _spec(empty_evals), "V007")


def test_v008_budget_sanity() -> None:
    def use_smaller_estimates(raw: dict[str, Any]) -> None:
        raw["nodes"][0]["cost_estimate"]["tokens"] = 100
        raw["nodes"][1]["cost_estimate"]["tokens"] = 100

    def exceed_budget(raw: dict[str, Any]) -> None:
        raw["budget_policy"]["pools"]["interactive"]["tokens"] = 1000
        raw["nodes"][0]["cost_estimate"]["tokens"] = 2000

    _assert_rule(_spec(), _spec(use_smaller_estimates), _spec(exceed_budget), "V008")


def test_v009_promotion_thresholds_sane() -> None:
    def widen_threshold(raw: dict[str, Any]) -> None:
        raw["fact_types"][0]["promote_auto"] = 10
        raw["fact_types"][0]["promote_review"] = 3

    def invert_threshold(raw: dict[str, Any]) -> None:
        raw["fact_types"][0]["promote_auto"] = 2
        raw["fact_types"][0]["promote_review"] = 5

    _assert_rule(_spec(), _spec(widen_threshold), _spec(invert_threshold), "V009")


def test_v010_elicitation_requires_node_permission() -> None:
    def allowed_elicitation_few_shot(raw: dict[str, Any]) -> None:
        raw["nodes"][0]["few_shots"] = [{"input": "ask", "output": {"status": "need_user_input"}}]
        raw["nodes"][0]["can_elicit"] = True

    def denied_elicitation_few_shot(raw: dict[str, Any]) -> None:
        raw["nodes"][1]["few_shots"] = [{"input": "ask", "output": {"status": "need_user_input"}}]
        raw["nodes"][1]["can_elicit"] = False

    _assert_rule(
        _spec(),
        _spec(allowed_elicitation_few_shot),
        _spec(denied_elicitation_few_shot),
        "V010",
    )
