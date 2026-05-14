"""Tests for the artifact Workspace."""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_intern.tools.workspace import Workspace


def test_put_and_get_roundtrip() -> None:
    ws = Workspace()
    df = pd.DataFrame({"x": [1, 2, 3]})
    ws.put("prices", df)
    assert ws.has("prices")
    pd.testing.assert_frame_equal(ws.get("prices"), df)


def test_missing_artifact_raises() -> None:
    ws = Workspace()
    with pytest.raises(KeyError):
        ws.get("nope")
    with pytest.raises(KeyError):
        ws.remove("nope")


def test_invalid_name_rejected() -> None:
    ws = Workspace()
    with pytest.raises(ValueError):
        ws.put("", 1)


def test_names_sorted_and_len() -> None:
    ws = Workspace()
    ws.put("b", 1)
    ws.put("a", 2)
    assert ws.names() == ["a", "b"]
    assert len(ws) == 2
    assert "a" in ws
