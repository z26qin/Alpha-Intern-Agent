"""File-based JSONL research memory store.

Each memory item is a single JSON object on its own line. Search is a
simple case-insensitive substring match against title, content, ticker,
and tags. Vector search will land in a later PR.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class MemoryItem(BaseModel):
    """A single research note."""

    id: str = Field(default_factory=_new_id)
    timestamp: str = Field(default_factory=_now_iso)
    memory_type: str = "note"
    ticker: Optional[str] = None
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ResearchMemoryStore:
    """JSONL-backed research memory store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def add_memory(
        self,
        title: str,
        content: str,
        *,
        memory_type: str = "note",
        ticker: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[dict] = None,
    ) -> MemoryItem:
        item = MemoryItem(
            title=title,
            content=content,
            memory_type=memory_type,
            ticker=ticker,
            tags=list(tags) if tags is not None else [],
            metadata=dict(metadata) if metadata is not None else {},
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(item.model_dump_json() + "\n")
        return item

    def _iter_items(self) -> Iterable[MemoryItem]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield MemoryItem.model_validate_json(line)

    def list_recent(self, limit: int = 10) -> list[MemoryItem]:
        items = list(self._iter_items())
        items.sort(key=lambda x: x.timestamp, reverse=True)
        return items[:limit]

    def search_memory(
        self,
        query: str,
        ticker: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        q = (query or "").strip().lower()
        tag_set = {t.lower() for t in tags} if tags else None

        results: list[MemoryItem] = []
        for item in self._iter_items():
            if ticker is not None and (item.ticker or "").lower() != ticker.lower():
                continue
            if tag_set is not None:
                item_tags = {t.lower() for t in item.tags}
                if not tag_set.issubset(item_tags):
                    continue
            if q:
                haystack = " ".join(
                    [
                        item.title or "",
                        item.content or "",
                        item.ticker or "",
                        " ".join(item.tags),
                    ]
                ).lower()
                if q not in haystack:
                    continue
            results.append(item)

        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]
