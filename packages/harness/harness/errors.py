"""Canonical error taxonomy for the harness kernel."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for errors that cross harness module boundaries."""


class InfraRetryable(HarnessError):
    """Transient infrastructure failure that retry machinery can handle."""


class FencingError(HarnessError):
    """A stale lease owner attempted to write after takeover."""


class PolicyDenied(HarnessError):
    """Policy rejected an attempted action."""


class SchemaViolation(HarnessError):
    """Structured model output or spec data failed schema validation."""


class BudgetExhausted(HarnessError):
    """A run exhausted its configured resource budget."""


class MissingCassette(HarnessError):
    """Replay mode could not find a required cassette entry."""


class ToolRejected(HarnessError):
    """A tool rejected a request in a planner-visible way."""


class SemanticFailure(HarnessError):
    """A node completed but could not satisfy its semantic goal."""


class ToolOutcomeUnknown(HarnessError):
    """A non-idempotent write may have reached the tool but no outcome is known."""


class ValidationFailed(HarnessError):
    """A spec, registry entry, or runtime binding failed validation."""

