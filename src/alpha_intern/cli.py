"""Typer CLI for AlphaInternAgent.

Just enough surface to prove the package is wired up. An LLM agent can
later drive the same primitives directly through the `tools/` package.
"""

from __future__ import annotations

from pathlib import Path
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


@app.command("agent")
def agent(
    goal: str = typer.Argument(..., help="The research goal to investigate."),
    max_steps: int = typer.Option(8, "--max-steps", help="Max planner steps before stopping."),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Per-call max output tokens."),
    model: Optional[str] = typer.Option(None, "--model", help="Anthropic model id."),
    memory_query: Optional[str] = typer.Option(
        None, "--memory-query", help="Override the memory-search query (defaults to goal)."
    ),
    memory_ticker: Optional[str] = typer.Option(
        None, "--memory-ticker", help="Only include memories for this ticker."
    ),
    run_log_path: Optional[Path] = typer.Option(
        None, "--run-log", help="Path to write the JSONL run log (default: data_dir/run.jsonl)."
    ),
    reflect: bool = typer.Option(
        False, "--reflect", help="After the run, ask the model to reflect on the trace and save a 'lesson' memory."
    ),
    seed_synthetic: bool = typer.Option(
        False,
        "--seed-synthetic",
        help=(
            "Before the agent runs, populate the workspace with a deterministic "
            "synthetic OHLCV panel saved under 'prices_raw' (handy first-run demo)."
        ),
    ),
) -> None:
    """Run the LLM agent on a research goal. Requires ANTHROPIC_API_KEY."""
    from alpha_intern.agent.loop import run_agent
    from alpha_intern.agent.provider import DEFAULT_MODEL, AnthropicProvider
    from alpha_intern.agent.run_log import RunLog
    from alpha_intern.tools import Workspace, get_default_registry
    from alpha_intern.tools.registry import ToolContext

    settings = get_settings()
    settings.ensure_dirs()

    provider = AnthropicProvider(model=model or DEFAULT_MODEL)
    registry = get_default_registry()

    log_path = run_log_path or (settings.data_dir / "run.jsonl")
    workspace = Workspace()
    memory = ResearchMemoryStore(settings.memory_path)
    skills = SkillRegistry(settings.skills_path)

    with RunLog(log_path) as log:
        ctx = ToolContext(
            workspace=workspace,
            memory=memory,
            skills=skills,
            run_log=log,
        )
        if seed_synthetic:
            seed_out = registry.dispatch(
                "load_synthetic_prices",
                inputs={"output_artifact": "prices_raw"},
                ctx=ctx,
            )
            typer.echo(
                f"Seeded workspace with synthetic prices: "
                f"{seed_out.n_rows} rows, {seed_out.n_tickers} tickers"
            )
        result = run_agent(
            goal=goal,
            provider=provider,
            registry=registry,
            ctx=ctx,
            max_steps=max_steps,
            max_tokens=max_tokens,
            memory_query=memory_query,
            memory_ticker=memory_ticker,
            reflect_at_end=reflect,
        )

    typer.echo("")
    typer.echo(f"Run {result.run_id} — stopped: {result.stopped_reason}")
    typer.echo(f"Steps used: {result.steps_used}, tool calls: {len(result.tool_calls)}")
    if result.tool_calls:
        typer.echo("Tools invoked:")
        for tc in result.tool_calls:
            status = "ok" if tc.get("ok") else f"ERROR ({tc.get('error')})"
            typer.echo(f"  step {tc['step']}: {tc['tool']} — {status}")
    if result.error:
        typer.echo(f"Error: {result.error}")
    typer.echo("")
    typer.echo("Final message:")
    typer.echo(result.final_text or "(no text)")

    if result.reflection is not None:
        ref = result.reflection
        typer.echo("")
        typer.echo("Reflection:")
        typer.echo(f"  memory_id: {ref.memory_id or '(not saved)'}")
        if ref.parse_error:
            typer.echo(f"  parse_error: {ref.parse_error}")
        typer.echo(f"  summary: {ref.payload.summary or '(empty)'}")
        if ref.payload.what_worked:
            typer.echo("  what_worked:")
            for item in ref.payload.what_worked:
                typer.echo(f"    - {item}")
        if ref.payload.what_failed:
            typer.echo("  what_failed:")
            for item in ref.payload.what_failed:
                typer.echo(f"    - {item}")
        if ref.payload.recommendations:
            typer.echo("  recommendations:")
            for item in ref.payload.recommendations:
                typer.echo(f"    - {item}")


if __name__ == "__main__":  # pragma: no cover
    app()
