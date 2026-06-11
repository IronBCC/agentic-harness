"""Pure event replay helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.types import Event, EventKind


@dataclass(frozen=True)
class RunState:
    """Minimal replayed state for M0 validation."""

    node_statuses: dict[str, str] = field(default_factory=dict)
    last_yields: dict[str, dict[str, object]] = field(default_factory=dict)
    spent_budget: int = 0
    finished: bool = False


def replay(events: list[Event]) -> RunState:
    """Fold ordered events into a deterministic run-state snapshot."""
    node_statuses: dict[str, str] = {}
    last_yields: dict[str, dict[str, object]] = {}
    spent_budget = 0
    finished = False

    for event in sorted(events, key=lambda item: item.seq):
        if event.kind == EventKind.node_started:
            node_statuses[event.node_id] = "started"
        elif event.kind == EventKind.yielded:
            node_statuses[event.node_id] = "yielded"
            last_yields[event.node_id] = dict(event.payload)
        elif event.kind == EventKind.llm_call:
            spent = event.payload.get("spent", 0)
            if isinstance(spent, int):
                spent_budget += spent
        elif event.kind == EventKind.run_finished:
            finished = True

    return RunState(
        node_statuses=node_statuses,
        last_yields=last_yields,
        spent_budget=spent_budget,
        finished=finished,
    )

