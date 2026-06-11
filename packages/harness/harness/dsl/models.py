"""Pydantic models for the AgentSpec DSL."""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

RequestClass = Literal["interactive", "background"]
NodeKind = Literal["planner", "router", "leaf", "tool_chain", "synthesizer"]
DecisionMode = Literal["micro", "freeform"]
SideEffect = Literal["pure", "read", "write"]
Idempotency = Literal["keyed", "none"]
Freshness = Literal["pure", "session", "volatile"]
AuthMode = Literal["service", "user_passthrough"]
Namespace = Literal["run", "session", "user", "cohort", "tenant"]


def _default_scope_allowed() -> list[Namespace]:
    return ["run"]


class StrictModel(BaseModel):
    """Base model for DSL records."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FactTypeDef(StrictModel):
    name: str
    schema_: dict[str, object] = Field(alias="schema")
    scope_allowed: list[Namespace] = Field(default_factory=_default_scope_allowed)
    promote_auto: float = 5.0
    promote_review: float = 2.0
    ttl_seconds: int | None = None
    confidence_decay: float | None = None


class ToolBinding(StrictModel):
    tool_id: str
    side_effect: SideEffect
    idempotency: Idempotency = "none"
    freshness: Freshness
    auth_mode: AuthMode
    requires_approval: bool = False
    emits_fact_types: list[str] = Field(default_factory=list)


class ModelBinding(StrictModel):
    name: str
    provider: str
    model: str
    params: dict[str, object] = Field(default_factory=dict)
    required_caps: list[str] = Field(default_factory=list)


class FewShot(StrictModel):
    input: dict[str, object] | str
    output: dict[str, object] | str


class RetryPolicy(StrictModel):
    max_attempts: int
    backoff_ms: int
    jitter: float


class CostEstimate(StrictModel):
    tokens: int
    usd: float
    wall_ms: int
    llm_calls: int


class CachePolicy(StrictModel):
    enabled: bool = False
    ttl_seconds: int | None = None


class NodeSpec(StrictModel):
    node_id: str
    kind: NodeKind
    model: str
    escalation: list[str] = Field(default_factory=list)
    capability_set: list[str] = Field(default_factory=list)
    max_context_tokens: int
    decision_mode: DecisionMode = "micro"
    prompt_template: str
    few_shots: list[FewShot] = Field(default_factory=list)
    output_schema: str | None = None
    retry: RetryPolicy
    cost_estimate: CostEstimate
    cache: CachePolicy = Field(default_factory=CachePolicy)
    speculative_prefetch: list[str] = Field(default_factory=list)
    can_elicit: bool = False


class Edge(StrictModel):
    from_node: str
    to_node: str
    condition: str | None = None


class Pool(StrictModel):
    tokens: int
    usd: float
    wall_ms: int
    llm_calls: int


class DegradationRules(StrictModel):
    on_yellow: str = "defer_optional"
    on_red: str = "wrap_up"


class ElicitationPolicy(StrictModel):
    max_questions_per_run: int = 3
    batch_window_ms: int = 500
    timeout_s: int = 900
    on_timeout: Literal["best_effort", "park", "fail"] = "best_effort"


class BudgetPolicy(StrictModel):
    pools: dict[RequestClass, Pool]
    lease_decay: float = 0.6
    pressure_thresholds: tuple[float, float] = (0.5, 0.15)
    degradation: DegradationRules
    max_depth_circuit_breaker: int = 12
    elicitation: ElicitationPolicy = Field(default_factory=ElicitationPolicy)


class PolicyRefs(StrictModel):
    tool_policy: str | None = None
    redaction_policy: str | None = None
    approval_policy: str | None = None


class EvalCase(StrictModel):
    case_id: str
    input: dict[str, object]
    expected: dict[str, object]
    tags: list[str] = Field(default_factory=list)


class AgentSpec(StrictModel):
    spec_id: str
    version: int
    description: str
    fact_types: list[FactTypeDef]
    tools: list[ToolBinding]
    models: list[ModelBinding]
    nodes: list[NodeSpec]
    edges: list[Edge]
    budget_policy: BudgetPolicy
    policies: PolicyRefs
    evals: list[EvalCase]


def loads_yaml(document: str) -> AgentSpec:
    """Parse YAML into an AgentSpec."""
    raw = yaml.safe_load(document)
    return AgentSpec.model_validate(raw)


def dumps_yaml(spec: AgentSpec) -> str:
    """Serialize an AgentSpec to normalized, byte-stable YAML."""
    raw = spec.model_dump(mode="json", by_alias=True, exclude_none=True)
    return yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
