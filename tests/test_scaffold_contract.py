from __future__ import annotations


def test_kernel_stub_modules_are_importable() -> None:
    import harness.errors
    import harness.protocols
    import harness.types
    import harness.util

    assert harness.errors.HarnessError.__name__ == "HarnessError"
    assert callable(harness.util.canon_hash)

