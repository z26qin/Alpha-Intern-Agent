"""Tests for the JSONL research memory store."""

from __future__ import annotations

from pathlib import Path

from alpha_intern.memory.store import ResearchMemoryStore


def test_add_and_list_recent(tmp_path: Path) -> None:
    store = ResearchMemoryStore(tmp_path / "mem.jsonl")
    store.add_memory(title="first", content="hello world", ticker="AAA")
    store.add_memory(
        title="second", content="momentum looks fine", ticker="BBB", tags=["momentum"]
    )

    items = store.list_recent(limit=10)
    assert len(items) == 2
    titles = {i.title for i in items}
    assert titles == {"first", "second"}


def test_search_by_query(tmp_path: Path) -> None:
    store = ResearchMemoryStore(tmp_path / "mem.jsonl")
    store.add_memory(title="momentum study", content="12-1 momentum on tech")
    store.add_memory(title="value study", content="cheap defensives")

    results = store.search_memory("momentum")
    assert len(results) == 1
    assert results[0].title == "momentum study"


def test_search_by_ticker_and_tags(tmp_path: Path) -> None:
    store = ResearchMemoryStore(tmp_path / "mem.jsonl")
    store.add_memory(
        title="apple notes", content="AAPL drift", ticker="AAPL", tags=["earnings"]
    )
    store.add_memory(
        title="msft notes", content="MSFT drift", ticker="MSFT", tags=["earnings"]
    )

    by_ticker = store.search_memory("", ticker="AAPL")
    assert len(by_ticker) == 1
    assert by_ticker[0].ticker == "AAPL"

    by_tag = store.search_memory("", tags=["earnings"])
    assert len(by_tag) == 2


def test_list_recent_respects_limit(tmp_path: Path) -> None:
    store = ResearchMemoryStore(tmp_path / "mem.jsonl")
    for i in range(5):
        store.add_memory(title=f"n{i}", content="x")

    assert len(store.list_recent(limit=2)) == 2
