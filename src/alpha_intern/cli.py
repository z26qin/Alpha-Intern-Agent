"""Typer CLI for AlphaInternAgent.

Just enough surface to prove the package is wired up. An LLM agent can
later drive the same primitives directly through the `tools/` package.
"""

from __future__ import annotations

from typing import Optional

import typer

from alpha_intern.config import get_settings
from alpha_intern.memory.skills import SkillRegistry
from alpha_intern.memory.store import ResearchMemoryStore

app = typer.Typer(
    add_completion=False,
    help="AlphaInternAgent — your stock-research intern.",
)


def _store() -> ResearchMemoryStore:
    s = get_settings()
    s.ensure_dirs()
    return ResearchMemoryStore(s.memory_path)


def _skills() -> SkillRegistry:
    s = get_settings()
    s.ensure_dirs()
    return SkillRegistry(s.skills_path)


@app.command("hello")
def hello() -> None:
    """Say hello and confirm the package is installed."""
    typer.echo("Hi, I'm Alpha Intern. Ready to do some (responsible) research.")


@app.command("skills-list")
def skills_list() -> None:
    """List registered research skills."""
    registry = _skills()
    skills = registry.list_skills()
    if not skills:
        typer.echo("(no skills registered)")
        return
    for skill in skills:
        typer.echo(f"- {skill.name}: {skill.description}")


@app.command("memory-add")
def memory_add(
    title: str = typer.Option(..., "--title", help="Short note title."),
    content: str = typer.Option(..., "--content", help="Note body."),
    ticker: Optional[str] = typer.Option(None, "--ticker", help="Optional ticker."),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="Comma-separated tags."
    ),
    memory_type: str = typer.Option("note", "--type", help="Memory type label."),
) -> None:
    """Append a research note to memory."""
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    tag_list = [t for t in tag_list if t]

    store = _store()
    item = store.add_memory(
        title=title,
        content=content,
        ticker=ticker,
        tags=tag_list,
        memory_type=memory_type,
    )
    typer.echo(f"Saved memory {item.id} ({item.title!r})")


@app.command("memory-search")
def memory_search(
    query: str = typer.Argument("", help="Substring to search for."),
    ticker: Optional[str] = typer.Option(None, "--ticker", help="Filter by ticker."),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="Comma-separated tags to require."
    ),
    limit: int = typer.Option(10, "--limit", help="Max results to show."),
) -> None:
    """Search research memory."""
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    store = _store()
    results = store.search_memory(
        query=query,
        ticker=ticker,
        tags=tag_list,
        limit=limit,
    )
    if not results:
        typer.echo("(no matches)")
        return
    for item in results:
        typer.echo(
            f"[{item.timestamp}] {item.id} {item.ticker or '-'} {item.title}"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
