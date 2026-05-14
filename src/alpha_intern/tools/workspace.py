"""In-process artifact store for tool pipelines.

Tools exchange DataFrames and fitted models by *name* via a Workspace.
This keeps tool inputs JSON-serializable (just artifact names + scalar
config), which is what a future LLM agent needs to call them.
"""

from __future__ import annotations

from typing import Any, Iterator


class Workspace:
    """A name-keyed in-memory store of artifacts (DataFrames, models, ...)."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def put(self, name: str, value: Any) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("Workspace artifact names must be non-empty strings")
        self._items[name] = value
        return name

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(f"Workspace has no artifact named {name!r}")
        return self._items[name]

    def has(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items.keys())

    def remove(self, name: str) -> None:
        if name not in self._items:
            raise KeyError(f"Workspace has no artifact named {name!r}")
        del self._items[name]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)
