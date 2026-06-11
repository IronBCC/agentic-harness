from __future__ import annotations

import tomllib
from pathlib import Path


def test_kernel_stub_modules_are_importable() -> None:
    import harness.errors
    import harness.protocols
    import harness.types
    import harness.util

    assert harness.errors.HarnessError.__name__ == "HarnessError"
    assert callable(harness.util.canon_hash)


def test_root_project_installs_harness_workspace_package() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "harness" in pyproject["project"]["dependencies"]
    assert pyproject["tool"]["uv"]["sources"]["harness"] == {"workspace": True}
