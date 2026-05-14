"""Smoke-test: every public module imports cleanly."""

from __future__ import annotations


def test_top_level_import() -> None:
    import alpha_intern

    assert hasattr(alpha_intern, "__version__")


def test_submodules_import() -> None:
    from alpha_intern import config  # noqa: F401
    from alpha_intern.backtest import metrics, walk_forward  # noqa: F401
    from alpha_intern.data import loader  # noqa: F401
    from alpha_intern.features import technical  # noqa: F401
    from alpha_intern.memory import skills, store  # noqa: F401
    from alpha_intern.models import signal_model  # noqa: F401
    from alpha_intern import tools  # noqa: F401


def test_cli_app_object_exists() -> None:
    from alpha_intern.cli import app

    assert app is not None
