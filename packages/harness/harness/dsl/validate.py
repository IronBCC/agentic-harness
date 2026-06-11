"""Static validation for AgentSpec documents."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from harness.dsl.models import AgentSpec, NodeSpec, ToolBinding

Risk = Literal["error", "warn"]
RegistryView = Mapping[str, object]
CapsView = Mapping[str, Mapping[str, object]]


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    path: str
    message: str
    risk: Risk


Rule = Callable[[AgentSpec, RegistryView, CapsView], list[Violation]]


def validate(spec: AgentSpec, registry_view: RegistryView, caps_view: CapsView) -> list[Violation]:
    """Return static validation violations for an AgentSpec."""
    violations: list[Violation] = []
    for rule in RULES:
        violations.extend(rule(spec, registry_view, caps_view))
    return violations


def v001_unreachable_node(
    spec: AgentSpec,
    _registry_view: RegistryView,
    _caps_view: CapsView,
) -> list[Violation]:
    if not spec.nodes:
        return []

    known_nodes = {node.node_id for node in spec.nodes}
    adjacency: dict[str, set[str]] = {node.node_id: set() for node in spec.nodes}
    for edge in spec.edges:
        if edge.from_node in known_nodes and edge.to_node in known_nodes:
            adjacency[edge.from_node].add(edge.to_node)

    reachable: set[str] = set()
    frontier = [spec.nodes[0].node_id]
    while frontier:
        node_id = frontier.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        frontier.extend(sorted(adjacency[node_id] - reachable))

    return [
        _violation("V001", f"nodes.{node.node_id}", "node is unreachable from entry node")
        for node in spec.nodes
        if node.node_id not in reachable
    ]


def v002_unresolved_tool_ref(
    spec: AgentSpec,
    registry_view: RegistryView,
    _caps_view: CapsView,
) -> list[Violation]:
    available = _registry_tools(registry_view)
    bound = {tool.tool_id for tool in spec.tools}
    violations: list[Violation] = []

    for tool in spec.tools:
        if tool.tool_id not in available:
            violations.append(
                _violation("V002", f"tools.{tool.tool_id}", "tool is not present in registry view")
            )

    for node in spec.nodes:
        for tool_id in [*node.capability_set, *node.speculative_prefetch]:
            if tool_id not in bound:
                violations.append(
                    _violation("V002", f"nodes.{node.node_id}.capability_set", "tool is not bound")
                )
            elif tool_id not in available:
                violations.append(
                    _violation(
                        "V002",
                        f"nodes.{node.node_id}.capability_set",
                        "bound tool is not present in registry view",
                    )
                )
    return violations


def v003_unresolved_model_ref(
    spec: AgentSpec,
    _registry_view: RegistryView,
    _caps_view: CapsView,
) -> list[Violation]:
    models = {model.name for model in spec.models}
    violations: list[Violation] = []
    for node in spec.nodes:
        refs = [node.model, *node.escalation]
        for ref in refs:
            if ref not in models:
                violations.append(
                    _violation("V003", f"nodes.{node.node_id}.model", "model ref is not declared")
                )
    return violations


def v004_model_caps_and_context(
    spec: AgentSpec,
    _registry_view: RegistryView,
    caps_view: CapsView,
) -> list[Violation]:
    violations: list[Violation] = []
    model_by_name = {model.name: model for model in spec.models}

    for model in spec.models:
        caps = set(_caps_list(caps_view.get(model.name, {})))
        missing = sorted(set(model.required_caps) - caps)
        if missing:
            violations.append(
                _violation(
                    "V004",
                    f"models.{model.name}.required_caps",
                    f"model lacks required capabilities: {', '.join(missing)}",
                )
            )

    for node in spec.nodes:
        if node.model not in model_by_name:
            continue
        max_context = _max_context(caps_view.get(node.model, {}))
        if max_context is not None and max_context < node.max_context_tokens:
            violations.append(
                _violation(
                    "V004",
                    f"nodes.{node.node_id}.max_context_tokens",
                    "node context budget exceeds model capability",
                )
            )
    return violations


def v005_write_none_constraints(
    spec: AgentSpec,
    _registry_view: RegistryView,
    _caps_view: CapsView,
) -> list[Violation]:
    violations: list[Violation] = []
    tools = {tool.tool_id: tool for tool in spec.tools}

    for tool in spec.tools:
        if (
            tool.side_effect == "write"
            and tool.idempotency == "none"
            and not tool.requires_approval
        ):
            violations.append(
                _violation(
                    "V005",
                    f"tools.{tool.tool_id}.requires_approval",
                    "write tool without keyed idempotency must require approval",
                )
            )

    for node in spec.nodes:
        if not _node_uses_non_idempotent_write(node, tools):
            continue
        if node.retry.max_attempts != 0:
            violations.append(
                _violation(
                    "V005",
                    f"nodes.{node.node_id}.retry.max_attempts",
                    "non-idempotent write tool nodes must not retry",
                )
            )
    return violations


def v006_undefined_fact_type(
    spec: AgentSpec,
    _registry_view: RegistryView,
    _caps_view: CapsView,
) -> list[Violation]:
    fact_types = {fact_type.name for fact_type in spec.fact_types}
    violations: list[Violation] = []

    for tool in spec.tools:
        for fact_type in tool.emits_fact_types:
            if fact_type not in fact_types:
                violations.append(
                    _violation(
                        "V006",
                        f"tools.{tool.tool_id}.emits_fact_types",
                        "tool emits undefined fact type",
                    )
                )

    for node in spec.nodes:
        if node.output_schema is not None and node.output_schema not in fact_types:
            violations.append(
                _violation(
                    "V006",
                    f"nodes.{node.node_id}.output_schema",
                    "node output schema references undefined fact type",
                )
            )
    return violations


def v007_empty_evals(
    spec: AgentSpec,
    _registry_view: RegistryView,
    _caps_view: CapsView,
) -> list[Violation]:
    if spec.evals:
        return []
    return [_violation("V007", "evals", "eval suite must not be empty")]


def v008_budget_sanity(
    spec: AgentSpec,
    _registry_view: RegistryView,
    _caps_view: CapsView,
) -> list[Violation]:
    total_tokens = sum(node.cost_estimate.tokens for node in spec.nodes)
    total_usd = sum(node.cost_estimate.usd for node in spec.nodes)
    total_wall_ms = sum(node.cost_estimate.wall_ms for node in spec.nodes)
    total_llm_calls = sum(node.cost_estimate.llm_calls for node in spec.nodes)
    totals = {
        "tokens": total_tokens,
        "usd": total_usd,
        "wall_ms": total_wall_ms,
        "llm_calls": total_llm_calls,
    }

    violations: list[Violation] = []
    for request_class, pool in spec.budget_policy.pools.items():
        pool_values = {
            "tokens": pool.tokens,
            "usd": pool.usd,
            "wall_ms": pool.wall_ms,
            "llm_calls": pool.llm_calls,
        }
        for dimension, total in totals.items():
            if total > pool_values[dimension]:
                violations.append(
                    _violation(
                        "V008",
                        f"budget_policy.pools.{request_class}.{dimension}",
                        "worst-case node estimates exceed pool",
                    )
                )
    return violations


def v009_promotion_thresholds(
    spec: AgentSpec,
    _registry_view: RegistryView,
    _caps_view: CapsView,
) -> list[Violation]:
    return [
        _violation(
            "V009",
            f"fact_types.{fact_type.name}",
            "promotion review threshold must be lower than auto threshold",
        )
        for fact_type in spec.fact_types
        if fact_type.promote_review >= fact_type.promote_auto
    ]


def v010_elicitation_permission(
    spec: AgentSpec,
    _registry_view: RegistryView,
    _caps_view: CapsView,
) -> list[Violation]:
    return [
        _violation(
            "V010",
            f"nodes.{node.node_id}.can_elicit",
            "node includes user elicitation examples without can_elicit",
        )
        for node in spec.nodes
        if _node_mentions_need_user_input(node) and not node.can_elicit
    ]


def _registry_tools(registry_view: RegistryView) -> set[str]:
    raw = registry_view.get("tools", set())
    if isinstance(raw, Mapping):
        return {str(tool_id) for tool_id in raw}
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, Iterable):
        return {str(tool_id) for tool_id in raw}
    return set()


def _caps_list(caps: Mapping[str, object]) -> list[str]:
    raw = caps.get("caps", [])
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Iterable):
        return [str(cap) for cap in raw]
    return []


def _node_uses_non_idempotent_write(
    node: NodeSpec,
    tools: Mapping[str, ToolBinding],
) -> bool:
    for tool_id in node.capability_set:
        tool = tools.get(tool_id)
        if tool is not None and tool.side_effect == "write" and tool.idempotency == "none":
            return True
    return False


def _max_context(caps: Mapping[str, object]) -> int | None:
    raw = caps.get("max_context")
    if isinstance(raw, int):
        return raw
    return None


def _node_mentions_need_user_input(node: NodeSpec) -> bool:
    for few_shot in node.few_shots:
        output = few_shot.output
        if isinstance(output, Mapping):
            if output.get("status") == "need_user_input":
                return True
        elif "need_user_input" in output:
            return True
    return node.output_schema == "need_user_input"


def _violation(code: str, path: str, message: str, risk: Risk = "error") -> Violation:
    return Violation(code=code, path=path, message=message, risk=risk)


RULES: tuple[Rule, ...] = (
    v001_unreachable_node,
    v002_unresolved_tool_ref,
    v003_unresolved_model_ref,
    v004_model_caps_and_context,
    v005_write_none_constraints,
    v006_undefined_fact_type,
    v007_empty_evals,
    v008_budget_sanity,
    v009_promotion_thresholds,
    v010_elicitation_permission,
)
