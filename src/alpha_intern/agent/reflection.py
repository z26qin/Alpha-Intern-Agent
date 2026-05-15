"""Post-run reflection: read a session's run-log trace, ask an LLM to
summarize what worked / what didn't, and persist the result as a
`memory_type="lesson"` note so future runs can retrieve it.

This is the "self-improving" leg of the Hermes-style agent. The
reflector itself is just one provider.generate(...) call — no tools,
no loop — so it's cheap and testable with a `ScriptedProvider`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from alpha_intern.agent.prompts import REFLECTION_SYSTEM_PROMPT
from alpha_intern.agent.provider import LLMProvider
from alpha_intern.agent.run_log import RunLog, RunLogEntry
from alpha_intern.memory.store import ResearchMemoryStore


class ReflectionPayload(BaseModel):
    """Structured fields the supervisor LLM is expected to return."""

    summary: str = ""
    what_worked: list[str] = Field(default_factory=list)
    what_failed: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    skill_suggestion: Optional[str] = None


@dataclass
class ReflectionResult:
    """End-of-reflection summary."""

    payload: ReflectionPayload
    memory_id: Optional[str]
    raw_text: str
    parse_error: Optional[str] = None


def _truncate_json(obj: object, max_chars: int = 240) -> str:
    if obj is None:
        return "null"
    try:
        s = json.dumps(obj, default=str)
    except (TypeError, ValueError):
        s = str(obj)
    return s if len(s) <= max_chars else s[: max_chars - 3] + "..."


def summarize_trace(
    entries: list[RunLogEntry], *, max_tool_calls: int = 25
) -> str:
    """Render a run-log trace into a compact text block for the supervisor."""
    lines: list[str] = []
    tool_calls_seen = 0
    for e in entries:
        if e.step_type == "run_start":
            lines.append(f"[run_start] run_id={e.run_id}")
        elif e.step_type == "plan":
            lines.append(f"[plan] {_truncate_json(e.metadata)}")
        elif e.step_type == "tool_call":
            tool_calls_seen += 1
            if tool_calls_seen <= max_tool_calls:
                lines.append(
                    f"[tool_call] {e.tool_name} "
                    f"input={_truncate_json(e.input)} "
                    f"output={_truncate_json(e.output)}"
                )
        elif e.step_type == "tool_error":
            lines.append(
                f"[tool_error] {e.tool_name} "
                f"input={_truncate_json(e.input)} error={e.error}"
            )
        elif e.step_type == "provider_error":
            lines.append(f"[provider_error] {_truncate_json(e.metadata)}")
        elif e.step_type == "note":
            content = (e.metadata or {}).get("content", "")
            lines.append(f"[note] {str(content)[:240]}")
        elif e.step_type == "run_end_summary":
            lines.append(f"[run_end_summary] {_truncate_json(e.metadata)}")

    if tool_calls_seen > max_tool_calls:
        skipped = tool_calls_seen - max_tool_calls
        lines.append(f"... and {skipped} more tool calls")
    return "\n".join(lines)


_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")


def _parse_json_response(text: str) -> tuple[Optional[dict], Optional[str]]:
    """Try hard to coax a JSON object out of the model's reply."""
    raw = (text or "").strip()
    if not raw:
        return None, "empty response"

    stripped = _FENCE_OPEN_RE.sub("", raw)
    stripped = _FENCE_CLOSE_RE.sub("", stripped).strip()
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass

    # Fallback: grab the largest {...} block.
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError as exc:
            return None, f"JSON decode error after extraction: {exc}"

    return None, "no JSON object found in response"


def _render_memory_body(
    payload: ReflectionPayload, goal: Optional[str]
) -> str:
    parts: list[str] = []
    if goal:
        parts.append(f"Goal: {goal}")
    if payload.summary:
        parts.append(payload.summary)
    if payload.what_worked:
        parts.append(
            "What worked:\n" + "\n".join(f"- {x}" for x in payload.what_worked)
        )
    if payload.what_failed:
        parts.append(
            "What failed:\n" + "\n".join(f"- {x}" for x in payload.what_failed)
        )
    if payload.recommendations:
        parts.append(
            "Recommendations:\n"
            + "\n".join(f"- {x}" for x in payload.recommendations)
        )
    if payload.skill_suggestion:
        parts.append(f"Skill suggestion: {payload.skill_suggestion}")
    return "\n\n".join(parts) or "(empty reflection)"


def _short_title(payload: ReflectionPayload) -> str:
    s = (payload.summary or "").strip().replace("\n", " ")
    if not s:
        return "Reflection (no summary)"
    return s if len(s) <= 80 else s[:77] + "..."


def reflect_on_run(
    *,
    provider: LLMProvider,
    run_log: Optional[RunLog] = None,
    entries: Optional[list[RunLogEntry]] = None,
    goal: Optional[str] = None,
    memory: Optional[ResearchMemoryStore] = None,
    extra_context: Optional[str] = None,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
    tags: Optional[list[str]] = None,
    write_memory: bool = True,
) -> ReflectionResult:
    """Read a run trace and produce a structured reflection.

    Either ``run_log`` (current-run entries are used) or ``entries`` must
    be provided. If ``memory`` is provided and ``write_memory`` is true,
    the reflection is persisted as a ``memory_type="lesson"`` note so
    future runs can find it via the standard memory search.
    """
    if entries is None:
        if run_log is None:
            raise ValueError("reflect_on_run requires run_log or entries")
        entries = run_log.current_run_entries()

    trace_text = summarize_trace(entries)
    body_parts: list[str] = []
    if goal:
        body_parts.append(f"## Goal\n{goal}")
    body_parts.append(f"## Run trace\n{trace_text or '(empty trace)'}")
    if extra_context:
        body_parts.append(f"## Additional context\n{extra_context}")
    body_parts.append(
        "Reflect on the run above. Reply with ONLY the JSON object described "
        "in the system prompt."
    )

    response = provider.generate(
        system=system_prompt or REFLECTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n\n".join(body_parts)}],
        tools=[],
        max_tokens=max_tokens,
    )

    raw_text = response.text or ""
    parsed, parse_error = _parse_json_response(raw_text)

    if parsed is None:
        payload = ReflectionPayload(summary=raw_text[:500] if raw_text else "(no response)")
    else:
        try:
            payload = ReflectionPayload.model_validate(parsed)
        except ValidationError as exc:
            payload = ReflectionPayload(summary=(raw_text or "")[:500])
            parse_error = f"validation error: {exc}"

    memory_id: Optional[str] = None
    if write_memory and memory is not None:
        item = memory.add_memory(
            title=_short_title(payload),
            content=_render_memory_body(payload, goal),
            memory_type="lesson",
            tags=list(tags) if tags else ["reflection", "auto"],
            metadata={
                "what_worked": payload.what_worked,
                "what_failed": payload.what_failed,
                "recommendations": payload.recommendations,
                "skill_suggestion": payload.skill_suggestion,
                "source_goal": goal,
                "trace_length": len(entries),
                "parse_error": parse_error,
            },
        )
        memory_id = item.id

    return ReflectionResult(
        payload=payload,
        memory_id=memory_id,
        raw_text=raw_text,
        parse_error=parse_error,
    )
