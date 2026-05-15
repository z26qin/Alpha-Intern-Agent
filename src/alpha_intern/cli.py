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

    # Write a run card so meta-reflect can learn from this run.
    from alpha_intern.agent.cards import build_card, write_card

    lessons_for_card: list[str] = []
    if result.reflection is not None:
        lessons_for_card = list(result.reflection.payload.recommendations or [])
    card = build_card(
        run_id=result.run_id or "unknown",
        goal=goal,
        final_text=result.final_text,
        stopped_reason=result.stopped_reason,
        steps_used=result.steps_used,
        tool_calls=result.tool_calls,
        lessons=lessons_for_card,
    )
    write_card(card, settings.data_dir)

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


@app.command("backlog-list")
def backlog_list() -> None:
    """Show pending research hypotheses."""
    from alpha_intern.agent.backlog import read_backlog

    settings = get_settings()
    settings.ensure_dirs()
    items = read_backlog(settings.data_dir)
    if not items:
        typer.echo("(backlog is empty — add items with `alpha-intern backlog-add`)")
        return
    for i, it in enumerate(items, 1):
        suffix = f" (tried {it.tried})" if it.tried else ""
        typer.echo(f"{i:>3}. {it.text}{suffix}")


@app.command("backlog-add")
def backlog_add(hypothesis: str = typer.Argument(..., help="One hypothesis to investigate.")) -> None:
    """Append a hypothesis to the backlog."""
    from alpha_intern.agent.backlog import append_backlog

    settings = get_settings()
    settings.ensure_dirs()
    append_backlog(hypothesis, settings.data_dir)
    typer.echo(f"Added: {hypothesis}")


@app.command("research")
def research(
    max_steps: int = typer.Option(8, "--max-steps"),
    max_tokens: int = typer.Option(4096, "--max-tokens"),
    model: Optional[str] = typer.Option(None, "--model"),
) -> None:
    """Pop the top backlog hypothesis, run the agent on it, write a run card."""
    from alpha_intern.agent.auto import run_auto_turn

    settings = get_settings()
    settings.ensure_dirs()
    run_log_path = settings.data_dir / "run.jsonl"

    # Force a research turn by seeding state so the chooser picks it.
    res = run_auto_turn(
        data_dir=settings.data_dir,
        run_log_path=run_log_path,
        daily_budget_usd=1e9,  # explicit single-shot — no budget gate
        max_steps=max_steps,
        max_tokens=max_tokens,
        model=model,
    )
    typer.echo(f"[{res.action}] {res.detail}")
    if res.card_path:
        typer.echo(f"card: {res.card_path}")


@app.command("meta-reflect")
def meta_reflect_cmd(
    model: Optional[str] = typer.Option(None, "--model"),
    n_cards: int = typer.Option(12, "--n-cards"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
) -> None:
    """Read recent run cards, propose new lessons/skills/hypotheses."""
    from alpha_intern.agent.meta_reflect import meta_reflect
    from alpha_intern.agent.provider import DEFAULT_MODEL, AnthropicProvider
    from alpha_intern.agent.run_log import RunLog

    settings = get_settings()
    settings.ensure_dirs()
    run_log_path = settings.data_dir / "run.jsonl"

    provider = AnthropicProvider(model=model or DEFAULT_MODEL)
    memory = _store()
    skills = _skills()

    with RunLog(run_log_path) as log:
        class _LP:
            def __init__(self, inner, log):
                self._inner = inner; self._log = log
            def generate(self, system, messages, tools, max_tokens=4096):
                import time as _t
                t0 = _t.monotonic()
                r = self._inner.generate(system, messages, tools, max_tokens)
                if r.usage:
                    self._log.log_llm_call(model=r.model or "", usage=r.usage,
                                           step=1, duration_s=_t.monotonic()-t0,
                                           stop_reason=r.stop_reason)
                return r

        res = meta_reflect(
            provider=_LP(provider, log),
            memory=memory,
            skills=skills,
            data_dir=settings.data_dir,
            n_cards=n_cards,
            max_tokens=max_tokens,
        )

    typer.echo(f"patterns ({len(res.payload.patterns)}):")
    for p in res.payload.patterns:
        typer.echo(f"  - {p}")
    typer.echo(f"lessons added: {len(res.written_memory_ids)}")
    typer.echo(f"skills added: {len(res.written_skill_names)}")
    typer.echo(f"hypotheses appended: {len(res.appended_hypotheses)}")
    if res.parse_error:
        typer.echo(f"parse_error: {res.parse_error}")


@app.command("auto")
def auto(
    budget: float = typer.Option(1.0, "--budget", help="Daily USD budget (default $1.00)."),
    max_steps: int = typer.Option(8, "--max-steps"),
    max_tokens: int = typer.Option(4096, "--max-tokens"),
    model: Optional[str] = typer.Option(None, "--model"),
) -> None:
    """One autonomous tick (research or meta-reflect). Designed for launchd/cron."""
    from alpha_intern.agent.auto import run_auto_turn

    settings = get_settings()
    settings.ensure_dirs()
    res = run_auto_turn(
        data_dir=settings.data_dir,
        run_log_path=settings.data_dir / "run.jsonl",
        daily_budget_usd=budget,
        max_steps=max_steps,
        max_tokens=max_tokens,
        model=model,
    )
    typer.echo(f"[{res.action}] {res.detail}  (today: ${res.cost_today_usd:.4f})")


@app.command("usage")
def usage(
    run_log_path: Optional[Path] = typer.Option(
        None, "--run-log", help="JSONL run log to read (default: data_dir/run.jsonl)."
    ),
    last: Optional[int] = typer.Option(
        None, "--last", help="Show only the N most recent runs."
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Filter to a single run_id."
    ),
) -> None:
    """Summarize Claude API token usage and estimated cost per run."""
    from alpha_intern.agent.usage import render_table, summarize_runs

    settings = get_settings()
    settings.ensure_dirs()
    path = run_log_path or (settings.data_dir / "run.jsonl")

    rows = summarize_runs(path)
    if run_id is not None:
        rows = [r for r in rows if r.run_id == run_id]
    if last is not None and last > 0:
        rows = rows[-last:]

    typer.echo(f"Run log: {path}")
    typer.echo(render_table(rows))


@app.command("chat")
def chat(
    max_steps: int = typer.Option(8, "--max-steps"),
    max_tokens: int = typer.Option(4096, "--max-tokens"),
    model: Optional[str] = typer.Option(None, "--model"),
    reflect: bool = typer.Option(False, "--reflect", help="Reflect after each turn."),
    seed_synthetic: bool = typer.Option(
        False, "--seed-synthetic", help="Seed workspace with synthetic prices at startup."
    ),
) -> None:
    """Interactive REPL. Type a research goal at each prompt; workspace, memory,
    and skills persist across turns. Type 'exit' or Ctrl-D to quit."""
    from alpha_intern.agent.loop import run_agent
    from alpha_intern.agent.provider import DEFAULT_MODEL, AnthropicProvider
    from alpha_intern.agent.run_log import RunLog
    from alpha_intern.tools import Workspace, get_default_registry
    from alpha_intern.tools.registry import ToolContext

    settings = get_settings()
    settings.ensure_dirs()

    provider = AnthropicProvider(model=model or DEFAULT_MODEL)
    registry = get_default_registry()
    workspace = Workspace()
    memory = ResearchMemoryStore(settings.memory_path)
    skills = SkillRegistry(settings.skills_path)
    log_path = settings.data_dir / "run.jsonl"

    typer.echo("Alpha Intern chat. Workspace persists across turns. Type 'exit' to quit.")

    with RunLog(log_path) as log:
        ctx = ToolContext(workspace=workspace, memory=memory, skills=skills, run_log=log)

        if seed_synthetic:
            seed_out = registry.dispatch(
                "load_synthetic_prices",
                inputs={"output_artifact": "prices_raw"},
                ctx=ctx,
            )
            typer.echo(
                f"Seeded synthetic prices: {seed_out.n_rows} rows, {seed_out.n_tickers} tickers"
            )

        turn = 0
        while True:
            try:
                goal = typer.prompt("\n>", prompt_suffix=" ").strip()
            except (EOFError, KeyboardInterrupt):
                typer.echo("")
                break
            if not goal:
                continue
            if goal.lower() in {"exit", "quit", ":q"}:
                break

            turn += 1
            try:
                result = run_agent(
                    goal=goal,
                    provider=provider,
                    registry=registry,
                    ctx=ctx,
                    max_steps=max_steps,
                    max_tokens=max_tokens,
                    reflect_at_end=reflect,
                )
            except Exception as exc:  # keep REPL alive
                typer.echo(f"[turn {turn}] error: {type(exc).__name__}: {exc}")
                continue

            from alpha_intern.agent.cards import build_card, write_card

            lessons: list[str] = []
            if result.reflection is not None:
                lessons = list(result.reflection.payload.recommendations or [])
            card = build_card(
                run_id=result.run_id or "unknown",
                goal=goal,
                final_text=result.final_text,
                stopped_reason=result.stopped_reason,
                steps_used=result.steps_used,
                tool_calls=result.tool_calls,
                lessons=lessons,
            )
            write_card(card, settings.data_dir)

            typer.echo("")
            typer.echo(result.final_text or "(no text)")
            tools_summary = ", ".join(tc["tool"] for tc in result.tool_calls) or "none"
            typer.echo(
                f"\n[turn {turn} · steps {result.steps_used} · "
                f"stop {result.stopped_reason} · tools: {tools_summary}]"
            )

    typer.echo("bye.")


if __name__ == "__main__":  # pragma: no cover
    app()
