from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest


@pytest.fixture
async def pg() -> AsyncIterator[str]:
    """Return the configured Postgres test DSN.

    M0-01 establishes the fixture contract. Later durability tickets replace this
    with template-database schema isolation.
    """
    dsn = os.getenv("HARNESS_TEST_DSN")
    if not dsn:
        pytest.skip("HARNESS_TEST_DSN is not set")
    yield dsn

