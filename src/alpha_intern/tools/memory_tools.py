"""Tool wrappers around alpha_intern.memory.store."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from alpha_intern.memory.store import MemoryItem
from alpha_intern.tools.registry import (
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
)


class AddMemoryIn(ToolInput):
    title: str = Field(..., description="Short title for the note.")
    content: str = Field(..., description="Body of the note.")
    ticker: Optional[str] = Field(default=None, description="Optional ticker the note is about.")
    tags: list[str] = Field(default_factory=list, description="Free-form tags.")
    memory_type: str = Field(default="note", description="Memory type label.")
    metadata: dict = Field(default_factory=dict, description="Arbitrary structured metadata.")


class AddMemoryOut(ToolOutput):
    id: str
    timestamp: str


class SearchMemoryIn(ToolInput):
    query: str = Field(default="", description="Substring to search title/content/tags/ticker for.")
    ticker: Optional[str] = Field(default=None)
    tags: Optional[list[str]] = Field(default=None)
    limit: int = Field(default=10, gt=0)


class SearchMemoryOut(ToolOutput):
    items: list[dict]


class ListRecentMemoryIn(ToolInput):
    limit: int = Field(default=10, gt=0)


def _require_memory(ctx: ToolContext) -> None:
    if ctx.memory is None:
        raise ToolError("This tool requires a ResearchMemoryStore in ToolContext")


def _serialize(item: MemoryItem) -> dict:
    return item.model_dump()


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        name="add_memory",
        description="Append a research note to the memory store.",
        input_model=AddMemoryIn,
        output_model=AddMemoryOut,
        tags=("memory",),
    )
    def _add(inp: AddMemoryIn, ctx: ToolContext) -> AddMemoryOut:
        _require_memory(ctx)
        item = ctx.memory.add_memory(
            title=inp.title,
            content=inp.content,
            ticker=inp.ticker,
            tags=inp.tags,
            memory_type=inp.memory_type,
            metadata=inp.metadata,
        )
        return AddMemoryOut(id=item.id, timestamp=item.timestamp)

    @registry.tool(
        name="search_memory",
        description="Substring-search the memory store, optionally filtered by ticker and tags.",
        input_model=SearchMemoryIn,
        output_model=SearchMemoryOut,
        tags=("memory",),
    )
    def _search(inp: SearchMemoryIn, ctx: ToolContext) -> SearchMemoryOut:
        _require_memory(ctx)
        items = ctx.memory.search_memory(
            query=inp.query,
            ticker=inp.ticker,
            tags=inp.tags,
            limit=inp.limit,
        )
        return SearchMemoryOut(items=[_serialize(i) for i in items])

    @registry.tool(
        name="list_recent_memories",
        description="Return the most recently added memory notes.",
        input_model=ListRecentMemoryIn,
        output_model=SearchMemoryOut,
        tags=("memory",),
    )
    def _list(inp: ListRecentMemoryIn, ctx: ToolContext) -> SearchMemoryOut:
        _require_memory(ctx)
        items = ctx.memory.list_recent(limit=inp.limit)
        return SearchMemoryOut(items=[_serialize(i) for i in items])
