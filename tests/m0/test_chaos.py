from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.chaos
@pytest.mark.asyncio
async def test_crash_matrix_writes_report_and_preserves_single_write_effect(pg: str) -> None:
    from tests.m0.chaos import run_crash_matrix

    report_path = Path("reports/m0/chaos.json")
    if report_path.exists():
        report_path.unlink()

    report = await run_crash_matrix(pg)

    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert {row["phase"] for row in report["matrix"]} == {
        "after_claim",
        "mid_execute",
        "pre_barrier",
        "post_barrier_pre_complete",
    }
    assert all(row["completed"] for row in report["matrix"])
    assert all(row["probe_effect_count"] == 1 for row in report["matrix"])
    assert all(row["stuck_tasks"] == 0 for row in report["matrix"])

