from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example() -> object:
    path = Path("examples/abc.py")
    spec = importlib.util.spec_from_file_location("abc_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_abc_example_builds_openai_compat_request() -> None:
    module = _load_example()

    request = module.build_request("gemma4")

    assert request.binding.provider == "openai_compat"
    assert request.binding.model == "gemma4"
    assert request.messages[0].role == "system"
    assert "Return only JSON" in request.messages[0].content


def test_abc_example_strips_markdown_json_fence() -> None:
    module = _load_example()

    assert module.clean_json_text('```json\n{"ok": true}\n```') == '{"ok": true}'
