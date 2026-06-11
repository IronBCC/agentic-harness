"""Single-process executor core."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import UUID

from harness.dsl.models import AgentSpec, NodeSpec
from harness.durability.postgres.backend import PostgresBackend
from harness.engine.retry import RetryDecision, plan_infra_retry, plan_schema_violation
from harness.errors import InfraRetryable, SchemaViolation
from harness.types import (
    Event,
    EventKind,
    NodeTask,
    NodeYield,
    RunInit,
    YieldStatus,
)
from harness.util import Clock, IdGen, SystemClock

NodeRunner = Callable[[NodeSpec, NodeTask], Awaitable[NodeYield]]


@dataclass
class Executor:
    """Claim tasks, execute nodes, and persist graph frontier progress."""

    backend: PostgresBackend
    spec: AgentSpec
    node_runner: NodeRunner
    idgen: IdGen
    worker: str
    clock: Clock = field(default_factory=SystemClock)
    batch_size: int = 20
    _locks: dict[UUID, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))

    async def seed_run(self, run: RunInit, input: dict[str, object]) -> None:
        """Create a run and enqueue entry nodes."""
        await self.backend.create_run(run)
        for node in self._entry_nodes():
            await self.backend.enqueue_task(
                task_id=self.idgen.new(),
                run_id=run.run_id,
                node_id=node.node_id,
                input=input,
            )

    async def run_once(self) -> int:
        """Claim one batch and execute claimed tasks concurrently."""
        tasks = await self.backend.claim(self.worker, self.batch_size)
        if not tasks:
            return 0
        await asyncio.gather(*(self._execute_task(task) for task in tasks))
        return len(tasks)

    async def run_until_idle(self) -> None:
        """Run batches until there is no immediately claimable work."""
        while await self.run_once():
            pass

    async def _execute_task(self, task: NodeTask) -> None:
        node = self._node_by_id(task.node_id)
        await self._append_event(
            task.run_id,
            node_id=node.node_id,
            kind=EventKind.node_started,
            payload={"attempt": task.attempt},
            idempotency_key=f"{task.task_id}:attempt:{task.attempt}:started",
        )
        try:
            node_yield = await self.node_runner(node, task)
        except InfraRetryable:
            await self._apply_retry_decision(task, plan_infra_retry(node, task, self.clock))
            return
        except SchemaViolation:
            await self._apply_retry_decision(task, plan_schema_violation(node, task, self.clock))
            return
        await self._append_event(
            task.run_id,
            node_id=node.node_id,
            kind=EventKind.yielded,
            payload=node_yield.model_dump(mode="json", by_alias=True, exclude_none=True),
            idempotency_key=f"{task.task_id}:yielded",
        )
        await self.backend.complete(task.task_id, terminal=node_yield.model_dump(mode="json"))
        await self._schedule_ready_successors(task.run_id)
        await self._finish_run_if_complete(task.run_id, node.node_id, node_yield)

    async def _apply_retry_decision(self, task: NodeTask, decision: RetryDecision) -> None:
        if decision.kind == "reschedule":
            if decision.available_at is None or decision.next_attempt is None:
                raise SchemaViolation("retry decision missing reschedule fields")
            await self.backend.reschedule(
                task.task_id,
                decision.available_at,
                decision.next_attempt,
                decision.next_input,
            )
            return

        if decision.node_yield is None:
            raise SchemaViolation("retry decision missing terminal yield")

        await self._append_event(
            task.run_id,
            node_id=task.node_id,
            kind=EventKind.yielded,
            payload=decision.node_yield.model_dump(mode="json", by_alias=True, exclude_none=True),
            idempotency_key=f"{task.task_id}:yielded",
        )
        await self.backend.complete(
            task.task_id,
            terminal=decision.node_yield.model_dump(mode="json"),
        )

    async def _schedule_ready_successors(self, run_id: UUID) -> None:
        async with self._locks[run_id]:
            tasks = await self.backend.list_tasks(run_id)
            existing_nodes = {task.node_id for task in tasks}
            done_nodes = {task.node_id for task in tasks if task.state == "done"}

            for node in self.spec.nodes:
                if node.node_id in existing_nodes:
                    continue
                predecessors = self._predecessors(node.node_id)
                if predecessors and predecessors <= done_nodes:
                    await self.backend.enqueue_task(
                        task_id=self.idgen.new(),
                        run_id=run_id,
                        node_id=node.node_id,
                        input={},
                    )

    async def _finish_run_if_complete(
        self,
        run_id: UUID,
        terminal_node_id: str,
        node_yield: NodeYield,
    ) -> None:
        if node_yield.status != YieldStatus.done:
            return
        async with self._locks[run_id]:
            tasks = await self.backend.list_tasks(run_id)
            done_nodes = {task.node_id for task in tasks if task.state == "done"}
            all_nodes = {node.node_id for node in self.spec.nodes}
            if done_nodes != all_nodes:
                return
            await self._append_event_locked(
                run_id,
                node_id="run",
                kind=EventKind.run_finished,
                payload={"status": "succeeded", "terminal_node": terminal_node_id},
                idempotency_key="run-finished",
            )
            await self.backend.update_run(
                run_id,
                status="succeeded",
                result={"terminal_node": terminal_node_id, "result_ref": node_yield.result_ref},
            )

    async def _append_event(
        self,
        run_id: UUID,
        *,
        node_id: str,
        kind: EventKind,
        payload: dict[str, object],
        idempotency_key: str,
    ) -> None:
        async with self._locks[run_id]:
            await self._append_event_locked(
                run_id,
                node_id=node_id,
                kind=kind,
                payload=payload,
                idempotency_key=idempotency_key,
            )

    async def _append_event_locked(
        self,
        run_id: UUID,
        *,
        node_id: str,
        kind: EventKind,
        payload: dict[str, object],
        idempotency_key: str,
    ) -> None:
        current = await self.backend.load(run_id)
        next_seq = max((event.seq for event in current.events), default=0) + 1
        await self.backend.append(
            run_id,
            [
                Event(
                    seq=next_seq,
                    node_id=node_id,
                    kind=kind,
                    payload=payload,
                    idempotency_key=idempotency_key,
                    barrier=True,
                )
            ],
        )

    def _entry_nodes(self) -> list[NodeSpec]:
        destinations = {edge.to_node for edge in self.spec.edges}
        return [node for node in self.spec.nodes if node.node_id not in destinations]

    def _predecessors(self, node_id: str) -> set[str]:
        return {edge.from_node for edge in self.spec.edges if edge.to_node == node_id}

    def _node_by_id(self, node_id: str) -> NodeSpec:
        for node in self.spec.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)
