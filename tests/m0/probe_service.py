from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ProbeService:
    """Tiny idempotent write probe used by the M0 reference runner."""

    _effects: Counter[str] = field(default_factory=Counter)

    async def write(self, idempotency_key: str) -> dict[str, object]:
        if self._effects[idempotency_key] == 0:
            self._effects[idempotency_key] += 1
        return {"ok": True, "idempotency_key": idempotency_key}

    def effects_for(self, idempotency_key: str) -> int:
        return self._effects[idempotency_key]

