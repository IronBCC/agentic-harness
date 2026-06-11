# Run Progress Observability Design

## Purpose

Agentic runs need an end-user-visible progress contract, not only debug traces.
The engine already records events, node tasks, budget snapshots, and pressure
state; this design defines the API shape that turns those internals into a
stable product surface.

Progress is deliberately framed as a mix of facts and estimates. In an agentic
graph, future work can expand through router decisions, fan-out, retries, and
spawned sub-runs, so the harness must never present percent complete or ETA as
absolute truth.

## Product Contract

Add a first-class run progress resource:

```http
GET /v1/runs/{id}/progress
```

The response is a `RunProgress` object:

```json
{
  "run_id": "uuid",
  "status": "running",
  "phase": "executing",
  "summary": "Searching inventory and checking policy constraints",
  "active_nodes": [
    {
      "node_id": "inventory_search",
      "label": "Search inventory",
      "state": "executing",
      "attempt": 0,
      "started_at": "2026-06-11T03:00:00Z"
    }
  ],
  "counts": {
    "known_total": 8,
    "completed": 3,
    "running": 2,
    "queued": 3,
    "waiting": 0,
    "retrying": 0,
    "failed": 0
  },
  "budget": {
    "tokens_remaining": 12000,
    "usd_remaining": 0.42,
    "wall_ms_remaining": 18000,
    "llm_calls_remaining": 7,
    "pressure": "green"
  },
  "estimate": {
    "known_graph_percent": 37.5,
    "remaining_is_estimate": true,
    "eta_ms": 14000,
    "basis": "known_frontier_and_cost_ema"
  },
  "waiting": null,
  "recent_events": []
}
```

Add an SSE event type on the existing stream:

```text
event: progress
data: <RunProgress JSON>
```

The stream remains the live channel for UI updates; `GET /progress` is the
snapshot endpoint for page load, polling fallback, and tests.

## Semantics

`status` is the durable run status from the `runs` row.

`phase` is UI-safe and derived from tasks/events:

- `queued`: run exists, no task currently claimed.
- `executing`: at least one task is executing or assembling.
- `waiting`: blocked on user input or approval.
- `retrying`: no task is executing and the next task is delayed by retry backoff.
- `wrapping_up`: budget or user request triggered wrap-up.
- `terminal`: succeeded, failed, cancelled, or exhausted.

`counts.known_total` is the current known graph size, not a promise that no new
nodes will appear. Dynamic spawn and fan-out can increase it.

`estimate.known_graph_percent` is computed only against known tasks:

```text
completed / max(known_total, 1) * 100
```

`estimate.eta_ms` is optional. It may be emitted only when the engine has enough
cost EMA data for the active node kinds. It must be marked as an estimate and
must be omitted rather than guessed when no credible basis exists.

`budget.pressure` is the governor's green/yellow/red state. Under M1 it can be
derived from static budget snapshots; under M3 it comes from the full budget
ledger and active leases.

`waiting` carries the actionable blocker when present:

```json
{
  "kind": "user_input",
  "questions": [{"id": "q1", "text": "Which account should I use?"}],
  "expires_at": "2026-06-11T03:15:00Z"
}
```

or:

```json
{
  "kind": "approval",
  "approval_id": "uuid",
  "tool_id": "crm.update",
  "reason": "write tool requires approval"
}
```

## Derivation

The progress reader is read-only. It derives state from:

- `runs.status`, `runs.budget`, and `runs.result`.
- `node_tasks.state`, `node_tasks.attempt`, `available_at`, and lease columns.
- `run_events` for node lifecycle transitions, labels, facts, errors, pressure
  changes, and recent activity.
- The budget ledger once M3 lands.

The progress reader must not call models, tools, or plugins.

## Event Model

The executor and lifecycle code should emit progress-relevant event payloads
using stable keys:

- `node_started`: `{label?, attempt}`
- `prompt_assembled`: `{label?, context_tokens?}`
- `llm_call`: `{model, prompt_tokens?, completion_tokens?, usd?}`
- `tool_call`: `{tool_id, side_effect, status}`
- `fact_emitted`: `{fact_type, confidence?}`
- `yielded`: `{status, confidence?, reason?}`
- `spawn_approved` / `spawn_denied`: `{count?, reason?}`
- `checkpoint`: `{before}`
- `run_finished`: `{status}`

The trace endpoint can expose full payloads; the progress endpoint should return
redacted, UI-safe summaries.

## Implementation Plan By Milestone

M1-05 to M1-06:

- Ensure executor/lifecycle emits enough node transition events to derive active
  node states.
- Keep labels optional and derived from `node_id` when no better label exists.

M1-15:

- Add `RunProgress` and `ProgressNode` Pydantic models.
- Add `GET /v1/runs/{id}/progress`.
- Add `progress` SSE event beside token, node transition, fact, and pressure
  events.
- Add tests for snapshot derivation and SSE ordering.

M3:

- Replace static budget-derived pressure with governor ledger pressure.
- Add ETA from cost EMAs.
- Add waiting payloads for HITL approvals and user elicitation.
- Add OTel attributes matching the progress fields.

## Testing

Unit tests:

- Running graph produces active node and count snapshots.
- Dynamic fan-out increases `known_total` without making prior percentages
  inconsistent.
- Waiting approval/user-input runs expose actionable `waiting` payloads.
- ETA omitted when no EMA basis exists.
- Terminal runs return `phase=terminal` and stable final counts.

API tests:

- `GET /progress` returns the same latest snapshot as the last SSE `progress`
  event.
- Progress events do not block token streaming.
- Progress payloads redact tool output and user PII.

Bench:

- SSE first-token delta remains under the existing `<15ms` gate.
- Progress snapshot derivation remains below `5ms` for a 100-node known graph.

## Non-Goals

- No exact percent-complete promise for dynamically expanding runs.
- No model-generated progress summaries in the kernel path.
- No UI implementation in the kernel package.
- No cross-run analytics dashboard in M1.

## Decisions

- `summary` is deterministic in M1. It is assembled from active node labels and
  phase, never from a model call.
- `/progress` does not include a compact run tree preview in M1. Tree shape stays
  on `/trace` until a UI implementation proves the need for a reduced view.
