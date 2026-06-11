from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.bench
@pytest.mark.asyncio
async def test_transition_overhead_bench_writes_report(pg: str) -> None:
    from tests.m0.bench import run_transition_bench

    report_path = Path("reports/m0/bench.json")
    if report_path.exists():
        report_path.unlink()

    report = await run_transition_bench(pg, transitions=250)

    assert report_path.exists()
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk == report
    assert report["transitions"] == 250
    assert report["p50_ms"] <= report["p95_ms"] <= report["p99_ms"]
    assert report["p95_ms"] > 0
