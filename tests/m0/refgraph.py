from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RefNode:
    node_id: str
    deps: tuple[str, ...] = ()
    write_key: str | None = None


@dataclass(frozen=True)
class RefGraph:
    nodes: tuple[RefNode, ...]
    by_id: dict[str, RefNode] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_id", {node.node_id: node for node in self.nodes})

    def ready(self, completed: set[str], scheduled: set[str]) -> list[RefNode]:
        return [
            node
            for node in self.nodes
            if node.node_id not in completed
            and node.node_id not in scheduled
            and all(dep in completed for dep in node.deps)
        ]


def build_reference_graph() -> RefGraph:
    """Return a 20-node mixed graph for M0 durability validation."""
    roots = tuple(RefNode(f"leaf-{index}") for index in range(1, 9))
    join = RefNode("join", deps=tuple(node.node_id for node in roots))
    chain = (
        RefNode("subagent-1", deps=("join",)),
        RefNode("subagent-2", deps=("subagent-1",)),
        RefNode("write", deps=("subagent-2",), write_key="reference-write"),
    )
    tail = tuple(
        RefNode(f"synth-{index}", deps=("write",) if index == 1 else (f"synth-{index - 1}",))
        for index in range(1, 9)
    )
    return RefGraph(nodes=roots + (join,) + chain + tail)

