# Agentic Harness

Self-hosted agentic execution harness for durable, declarative agent workflows.

The project is intentionally early-stage. Current implementation covers:

- Postgres-backed run/event durability.
- Postgres task queue with leases and `SKIP LOCKED` claims.
- Event replay helpers.
- M0 reference graph, benchmark, and deterministic chaos simulation.
- M1 AgentSpec DSL models with strict Pydantic validation and YAML round-trips.
- M1 executor frontier loop, node lifecycle/yield parsing, and retry/escalation planner.

## Development

Install dependencies:

```bash
uv sync
```

Start the local Postgres test database:

```bash
docker compose -f docker-compose.dev.yml up -d
```

Run checks:

```bash
HARNESS_TEST_DSN=postgresql://harness:harness@localhost:55432/harness uv run pytest -q
uv run ruff check .
uv run mypy --strict packages/harness/harness
```

## Status

M0 durability validation is green. The latest local transition benchmark measured
`p95=2.34ms` for the pooled Postgres path; see `docs/M0-REPORT.md`.

Next planned ticket: M1-08 model gateway core + capabilities registry.

Progress/observability design for the future API lives in
`docs/superpowers/specs/2026-06-11-run-progress-observability-design.md`.
