"""Autonomous research turn: alternates research ↔ meta-reflect under a
daily $ budget. Designed to be invoked by launchd / cron.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from alpha_intern.agent.backlog import mark_attempt, pop_next, read_backlog
from alpha_intern.agent.cards import build_card, write_card
from alpha_intern.agent.usage import summarize_runs


@dataclass
class AutoResult:
    action: str  # "research" | "meta_reflect" | "skipped"
    detail: str
    cost_today_usd: float
    card_path: Optional[str] = None


def _state_path(data_dir: Path) -> Path:
    return data_dir / "auto_state.json"


def _read_state(data_dir: Path) -> dict:
    p = _state_path(data_dir)
    if not p.exists():
        return {"turn": 0, "last_action": None}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"turn": 0, "last_action": None}


def _write_state(state: dict, data_dir: Path) -> None:
    _state_path(data_dir).write_text(json.dumps(state, indent=2))


def cost_today_usd(run_log_path: Path) -> float:
    """Sum estimated cost of runs started today (UTC)."""
    today = datetime.now(timezone.utc).date().isoformat()
    total = 0.0
    for r in summarize_runs(run_log_path):
        if r.started.startswith(today):
            c = r.estimated_cost_usd()
            if c is not None:
                total += c
    return total


def _choose_action(state: dict) -> str:
    """3 research turns, then 1 meta-reflect, repeat."""
    return "meta_reflect" if (state["turn"] % 4 == 3) else "research"


def run_auto_turn(
    *,
    data_dir: Path,
    run_log_path: Path,
    daily_budget_usd: float,
    max_steps: int,
    max_tokens: int,
    model: Optional[str] = None,
) -> AutoResult:
    """One scheduled tick. Imports providers lazily so the module is
    importable without the [agent] extra."""
    spent = cost_today_usd(run_log_path)
    if spent >= daily_budget_usd:
        return AutoResult(
            action="skipped",
            detail=f"daily budget hit: ${spent:.4f} >= ${daily_budget_usd:.2f}",
            cost_today_usd=spent,
        )

    state = _read_state(data_dir)
    action = _choose_action(state)

    if action == "research":
        item = pop_next(data_dir)
        if item is None:
            action = "meta_reflect"  # nothing to do; fall back to learning

    from alpha_intern.agent.provider import DEFAULT_MODEL, AnthropicProvider
    from alpha_intern.agent.run_log import RunLog
    from alpha_intern.memory.skills import SkillRegistry
    from alpha_intern.memory.store import ResearchMemoryStore
    from alpha_intern.tools import Workspace, get_default_registry
    from alpha_intern.tools.registry import ToolContext

    provider = AnthropicProvider(model=model or DEFAULT_MODEL)
    memory = ResearchMemoryStore(data_dir / "memory.jsonl")
    skills = SkillRegistry(data_dir / "skills.json")

    if action == "research":
        from alpha_intern.agent.loop import run_agent

        registry = get_default_registry()
        workspace = Workspace()
        with RunLog(run_log_path) as log:
            ctx = ToolContext(workspace=workspace, memory=memory, skills=skills, run_log=log)
            try:
                result = run_agent(
                    goal=item.text,
                    provider=provider,
                    registry=registry,
                    ctx=ctx,
                    max_steps=max_steps,
                    max_tokens=max_tokens,
                    reflect_at_end=True,
                )
                success = result.stopped_reason in {"end_turn"}
            except Exception as exc:
                success = False
                result = None
                detail = f"error: {type(exc).__name__}: {exc}"

        if result is not None:
            lessons: list[str] = []
            if result.reflection is not None:
                lessons = list(result.reflection.payload.recommendations or [])
            card = build_card(
                run_id=result.run_id or "unknown",
                goal=item.text,
                final_text=result.final_text,
                stopped_reason=result.stopped_reason,
                steps_used=result.steps_used,
                tool_calls=result.tool_calls,
                lessons=lessons,
            )
            card_path = write_card(card, data_dir)
            mark_attempt(item.text, data_dir, success=success)
            detail = f"ran '{item.text[:60]}' → {result.stopped_reason}"
            state["turn"] += 1
            state["last_action"] = "research"
            _write_state(state, data_dir)
            return AutoResult(
                action="research",
                detail=detail,
                cost_today_usd=cost_today_usd(run_log_path),
                card_path=str(card_path),
            )

        mark_attempt(item.text, data_dir, success=False)
        state["turn"] += 1
        state["last_action"] = "research"
        _write_state(state, data_dir)
        return AutoResult(
            action="research",
            detail=detail,
            cost_today_usd=cost_today_usd(run_log_path),
        )

    # meta_reflect
    from alpha_intern.agent.meta_reflect import meta_reflect

    with RunLog(run_log_path) as log:
        # the meta call goes through the provider directly; we still want it
        # in the run log so usage gets billed against today's budget.
        from alpha_intern.agent.provider import LLMResponse  # noqa: F401  (typing only)

        # Wrap the provider so its single generate() also writes an llm_call.
        class _LoggingProvider:
            def __init__(self, inner, log):
                self._inner = inner
                self._log = log

            def generate(self, system, messages, tools, max_tokens=4096):
                import time as _t

                t0 = _t.monotonic()
                r = self._inner.generate(system, messages, tools, max_tokens)
                if r.usage:
                    self._log.log_llm_call(
                        model=r.model or "",
                        usage=r.usage,
                        step=1,
                        duration_s=_t.monotonic() - t0,
                        stop_reason=r.stop_reason,
                    )
                return r

        result = meta_reflect(
            provider=_LoggingProvider(provider, log),
            memory=memory,
            skills=skills,
            data_dir=data_dir,
        )

    state["turn"] += 1
    state["last_action"] = "meta_reflect"
    _write_state(state, data_dir)
    return AutoResult(
        action="meta_reflect",
        detail=(
            f"patterns={len(result.payload.patterns)} "
            f"lessons+{len(result.written_memory_ids)} "
            f"skills+{len(result.written_skill_names)} "
            f"hyps+{len(result.appended_hypotheses)}"
        ),
        cost_today_usd=cost_today_usd(run_log_path),
    )
