"""Minimal LLM agent loop.

Drives a single research session:
    1. Assemble context from goal + memory + skills + workspace.
    2. Call provider with tool schemas.
    3. If the response contains tool_use blocks, run them via the
       `ToolRegistry`, append tool_result messages, loop.
    4. If the response has no tool_use, terminate ("end_turn").
    5. If `max_steps` is hit first, terminate ("step_budget").

Every dispatch is automatically traced via `ToolContext.run_log`. The
loop itself also logs `plan` and `run_end_summary` events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from alpha_intern.agent.context import AssembledContext, assemble_context
from alpha_intern.agent.provider import LLMProvider, LLMResponse
from alpha_intern.tools.registry import ToolContext, ToolRegistry


@dataclass
class AgentResult:
    """End-of-run summary."""

    run_id: Optional[str]
    final_text: str
    steps_used: int
    stopped_reason: str  # "end_turn" | "step_budget" | "error"
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    reflection: Optional[Any] = None  # alpha_intern.agent.reflection.ReflectionResult


def _tools_for_provider(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Render registry tools into Anthropic tool-spec shape."""
    out = []
    for tool in registry.list_tools():
        out.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_model.model_json_schema(),
            }
        )
    return out


def _assistant_content(response: LLMResponse) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if response.text:
        blocks.append({"type": "text", "text": response.text})
    for tu in response.tool_uses:
        blocks.append(
            {
                "type": "tool_use",
                "id": tu.id,
                "name": tu.name,
                "input": tu.input,
            }
        )
    return blocks


def _stringify_tool_result(payload: Any) -> str:
    try:
        return json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return str(payload)


def run_agent(
    goal: str,
    *,
    provider: LLMProvider,
    registry: ToolRegistry,
    ctx: ToolContext,
    max_steps: int = 8,
    max_tokens: int = 4096,
    memory_query: Optional[str] = None,
    memory_ticker: Optional[str] = None,
    memory_tags: Optional[list[str]] = None,
    system_prompt: Optional[str] = None,
    assembled: Optional[AssembledContext] = None,
    reflect_at_end: bool = False,
    reflection_provider: Optional[LLMProvider] = None,
    reflection_max_tokens: int = 1024,
) -> AgentResult:
    """Run a single agent session and return the result."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    if assembled is None:
        assembled = assemble_context(
            goal=goal,
            memory=ctx.memory,
            skills=ctx.skills,
            workspace=ctx.workspace,
            memory_query=memory_query,
            memory_ticker=memory_ticker,
            memory_tags=memory_tags,
            system_prompt=system_prompt,
        )

    tool_specs = _tools_for_provider(registry)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": assembled.initial_user_message}
    ]

    if ctx.run_log is not None:
        ctx.run_log.log_event(
            "plan",
            goal=goal,
            n_tools=len(tool_specs),
            available_skills=assembled.skill_names,
            relevant_memory_ids=assembled.memory_ids,
        )

    final_text = ""
    stopped_reason = "step_budget"
    error: Optional[str] = None
    tool_call_records: list[dict[str, Any]] = []
    steps_used = 0

    for step_idx in range(max_steps):
        steps_used = step_idx + 1
        import time as _time

        t0 = _time.monotonic()
        try:
            response = provider.generate(
                system=assembled.system,
                messages=messages,
                tools=tool_specs,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            stopped_reason = "error"
            error = f"{type(exc).__name__}: {exc}"
            if ctx.run_log is not None:
                ctx.run_log.log_event("provider_error", error=error)
            break

        if ctx.run_log is not None and response.usage:
            ctx.run_log.log_llm_call(
                model=response.model or "",
                usage=response.usage,
                step=steps_used,
                duration_s=_time.monotonic() - t0,
                stop_reason=response.stop_reason,
            )

        messages.append({"role": "assistant", "content": _assistant_content(response)})

        if response.text:
            final_text = response.text

        if not response.tool_uses:
            stopped_reason = "end_turn"
            break

        tool_results: list[dict[str, Any]] = []
        for tu in response.tool_uses:
            record: dict[str, Any] = {
                "step": steps_used,
                "tool": tu.name,
                "input": tu.input,
                "ok": False,
            }
            try:
                out = registry.dispatch(tu.name, inputs=tu.input, ctx=ctx)
                payload = out.model_dump()
                record["ok"] = True
                record["output"] = payload
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": _stringify_tool_result(payload),
                    }
                )
            except Exception as exc:
                err_text = f"{type(exc).__name__}: {exc}"
                record["error"] = err_text
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "is_error": True,
                        "content": err_text,
                    }
                )
            tool_call_records.append(record)

        messages.append({"role": "user", "content": tool_results})

    if ctx.run_log is not None:
        ctx.run_log.log_event(
            "run_end_summary",
            stopped_reason=stopped_reason,
            steps_used=steps_used,
            n_tool_calls=len(tool_call_records),
            final_text=final_text[:500],
            error=error,
        )

    reflection_result = None
    if reflect_at_end:
        from alpha_intern.agent.reflection import reflect_on_run

        reflect_provider = reflection_provider or provider
        try:
            entries = (
                ctx.run_log.current_run_entries() if ctx.run_log is not None else []
            )
            reflection_result = reflect_on_run(
                provider=reflect_provider,
                entries=entries,
                goal=goal,
                memory=ctx.memory,
                max_tokens=reflection_max_tokens,
            )
            if ctx.run_log is not None:
                ctx.run_log.log_event(
                    "note",
                    content=f"reflection memory_id={reflection_result.memory_id} "
                    f"parse_error={reflection_result.parse_error}",
                    source="reflection",
                )
        except Exception as exc:
            if ctx.run_log is not None:
                ctx.run_log.log_event(
                    "note",
                    content=f"reflection failed: {type(exc).__name__}: {exc}",
                    source="reflection",
                )

    return AgentResult(
        run_id=getattr(ctx.run_log, "run_id", None),
        final_text=final_text,
        steps_used=steps_used,
        stopped_reason=stopped_reason,
        messages=messages,
        tool_calls=tool_call_records,
        error=error,
        reflection=reflection_result,
    )
