"""Deterministic utility primitives."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

import orjson


def canon_hash(value: object) -> str:
    """Return the canonical sha256 hash for a JSON-serializable value."""
    payload = orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


class IdGen(Protocol):
    """UUID source injected into runtime components."""

    def new(self) -> UUID:
        """Return a new UUID."""


class Clock(Protocol):
    """Clock injected into runtime components."""

    def now(self) -> datetime:
        """Return the current timezone-aware timestamp."""


@dataclass(frozen=True)
class Uuid4IdGen:
    """Production UUID generator."""

    def new(self) -> UUID:
        return uuid.uuid4()


@dataclass(frozen=True)
class SystemClock:
    """Production wall clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass
class SequentialIdGen:
    """Deterministic UUID generator for tests."""

    counter: int = 0

    def new(self) -> UUID:
        self.counter += 1
        return UUID(int=self.counter)


@dataclass
class FrozenClock:
    """Deterministic, manually-steppable clock for tests."""

    current: datetime

    def now(self) -> datetime:
        return self.current

    def set(self, value: datetime) -> None:
        self.current = value


def canonical_json(value: Mapping[str, object] | list[object] | object) -> bytes:
    """Serialize a JSON-compatible value using the harness canonical form."""
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)

