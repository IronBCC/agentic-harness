from __future__ import annotations

import asyncio
import os
from uuid import uuid4

from harness.durability.postgres.migrate import apply

from examples.scenario_a.runner import run_replay


async def main() -> None:
    dsn = os.environ["HARNESS_TEST_DSN"]
    await apply(dsn)
    loaded, facts = await run_replay(dsn, run_id=uuid4())
    print({"run_id": str(loaded.run_id), "status": loaded.status, "facts": facts})


if __name__ == "__main__":
    asyncio.run(main())
