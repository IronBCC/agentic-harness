from __future__ import annotations


def fixture_tools() -> list[dict[str, object]]:
    return [
        {
            "name": "ticket.lookup",
            "description": "Look up support ticket.",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "kb.search",
            "description": "Search support knowledge base.",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "ticket.update",
            "description": "Update support ticket.",
            "inputSchema": {"type": "object"},
            "annotations": {"side_effect": "write", "idempotency": "keyed"},
        },
    ]
