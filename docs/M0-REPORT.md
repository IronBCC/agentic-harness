# M0 Durability Validation Report

Date: 2026-06-11

## Summary

M0 validated the Postgres event-log and queue design enough to continue into M1.
The first benchmark missed the transition-overhead budget because hot paths opened
fresh Postgres connections per operation; shared asyncpg pool support brought the
measured transition path under budget.

## Results

| Metric | Target | Measured | Status |
|---|---:|---:|---|
| Transition overhead p95 | < 10 ms | 2.34 ms | PASS |
| Transition overhead p50 | n/a | 1.98 ms | info |
| Transition overhead p99 | n/a | 2.79 ms | info |
| Claim throughput | >= 2,000 tasks/s | 7,901.63 tasks/s | PASS |
| Crash matrix | all phases complete | 4/4 complete | PASS |
| Probe write effects | exactly 1 per phase | 1 in every phase | PASS |
| Stuck tasks after chaos | 0 | 0 | PASS |

Source reports:

- `reports/m0/bench.json`
- `reports/m0/claim-throughput.json`
- `reports/m0/chaos.json`

## Crash Matrix

| Phase | Completed | Probe effects | Stuck tasks |
|---|---:|---:|---:|
| after_claim | true | 1 | 0 |
| mid_execute | true | 1 | 0 |
| pre_barrier | true | 1 | 0 |
| post_barrier_pre_complete | true | 1 | 0 |

## Protocol Decision

`DurabilityBackend` protocol signatures are usable for the next design pass. The
current queue and event log now support caller-owned asyncpg pools while preserving
the simpler DSN-only path for tests and tools. Remaining hardening before production
use:

1. Replace the deterministic chaos simulation with a subprocess crash harness.
2. Batch transition writes more aggressively where barrier semantics allow it.
3. Add longer soak coverage with pool sizing and Postgres connection-limit checks.

## Decision

GO to M1. The Postgres design passes the M0 correctness checks and the p95
transition-overhead budget.
