"""Cross-run reflection: read recent run cards and propose memory/skill
updates and new backlog hypotheses.

This is a *single* provider call with no tools. Output is structured
JSON. Dedup against existing memories and skills happens here, not in
the LLM — the model proposes, we filter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from alpha_intern.agent.cards import RunCard, read_cards
from alpha_intern.agent.provider import LLMProvider
from alpha_intern.memory.skills import SkillRegistry
from alpha_intern.memory.store import ResearchMemoryStore


META_SYSTEM = """You are a research-team lead reviewing the recent work of an autonomous
research intern. You will be shown short cards describing the intern's
last several runs (goal, tools used, errors, metrics, lessons).

Your job is to find recurring patterns — not narrate single runs.
Propose only items the intern would not already have written down.
Be concise. Output ONLY a single JSON object with this shape:

{
  "patterns": ["short observation", ...],
  "new_lessons": [
    {"title": "short", "content": "1-3 sentence rule with WHY and WHEN", "tags": ["..."]},
    ...
  ],
  "new_skills": [
    {"name": "snake_case", "description": "1 sentence", "tags": ["..."]},
    ...
  ],
  "new_hypotheses": ["short researchable question", ...]
}

Keep each list <= 5 items. Empty lists are fine. No prose outside the
JSON object."""


class _LessonItem(BaseModel):
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)


class _SkillItem(BaseModel):
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)


class MetaReflectionPayload(BaseModel):
    patterns: list[str] = Field(default_factory=list)
    new_lessons: list[_LessonItem] = Field(default_factory=list)
    new_skills: list[_SkillItem] = Field(default_factory=list)
    new_hypotheses: list[str] = Field(default_factory=list)


@dataclass
class MetaReflectionResult:
    payload: MetaReflectionPayload
    written_memory_ids: list[str] = field(default_factory=list)
    written_skill_names: list[str] = field(default_factory=list)
    appended_hypotheses: list[str] = field(default_factory=list)
    raw_text: str = ""
    parse_error: Optional[str] = None


def _render_cards(cards: list[RunCard]) -> str:
    lines: list[str] = []
    for c in cards:
        lines.append(
            f"- run {c.run_id} ({c.created_at[:10]}): goal={c.goal!r} "
            f"stopped={c.stopped_reason} steps={c.steps_used} "
            f"tools=[{','.join(c.tools_used)}] errors={c.error_count}"
        )
        if c.metrics:
            metric_str = ", ".join(f"{k}={v:.3f}" for k, v in c.metrics.items())
            lines.append(f"    metrics: {metric_str}")
        if c.lessons:
            for ln in c.lessons:
                lines.append(f"    lesson: {ln}")
        if c.final_text:
            lines.append(f"    final: {c.final_text[:200]}")
    return "\n".join(lines)


def _render_existing(memory: ResearchMemoryStore, skills: SkillRegistry) -> str:
    mem_lines: list[str] = []
    for m in memory.search_memory(query="", limit=50):
        mem_lines.append(f"- ({m.memory_type}) {m.title}")
    sk_lines = [f"- {s.name}: {s.description}" for s in skills.list_skills()]
    return (
        "Existing memories:\n" + ("\n".join(mem_lines) or "(none)") +
        "\n\nExisting skills:\n" + ("\n".join(sk_lines) or "(none)")
    )


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _parse_payload(text: str) -> tuple[MetaReflectionPayload, Optional[str]]:
    m = _JSON_BLOCK_RE.search(text)
    raw = m.group(0) if m else text
    try:
        return MetaReflectionPayload.model_validate_json(raw), None
    except ValidationError as exc:
        return MetaReflectionPayload(), str(exc)
    except json.JSONDecodeError as exc:
        return MetaReflectionPayload(), str(exc)


def _dedup_lesson(item: _LessonItem, memory: ResearchMemoryStore) -> bool:
    """Return True if a near-duplicate already exists."""
    title_low = item.title.lower().strip()
    for m in memory.search_memory(query="", limit=200):
        if m.title.lower().strip() == title_low:
            return True
    return False


def _dedup_skill(item: _SkillItem, skills: SkillRegistry) -> bool:
    for s in skills.list_skills():
        if s.name == item.name:
            return True
    return False


def meta_reflect(
    *,
    provider: LLMProvider,
    memory: ResearchMemoryStore,
    skills: SkillRegistry,
    data_dir: Path,
    n_cards: int = 12,
    max_tokens: int = 1024,
) -> MetaReflectionResult:
    cards = read_cards(data_dir, limit=n_cards)
    if not cards:
        return MetaReflectionResult(payload=MetaReflectionPayload(), raw_text="(no cards)")

    user = (
        "Recent runs:\n"
        + _render_cards(cards)
        + "\n\n"
        + _render_existing(memory, skills)
        + "\n\nReturn JSON only."
    )
    resp = provider.generate(
        system=META_SYSTEM,
        messages=[{"role": "user", "content": user}],
        tools=[],
        max_tokens=max_tokens,
    )
    payload, err = _parse_payload(resp.text)
    result = MetaReflectionResult(payload=payload, raw_text=resp.text, parse_error=err)

    # Apply with dedup.
    from alpha_intern.agent.backlog import append_backlog
    from alpha_intern.memory.skills import ResearchSkill

    for lesson in payload.new_lessons:
        if _dedup_lesson(lesson, memory):
            continue
        item = memory.add_memory(
            title=lesson.title,
            content=lesson.content,
            tags=list(lesson.tags) + ["meta_reflection"],
            memory_type="lesson",
        )
        result.written_memory_ids.append(item.id)

    for sk in payload.new_skills:
        if _dedup_skill(sk, skills):
            continue
        skills.add_skill(
            ResearchSkill(
                name=sk.name,
                description=sk.description,
                tags=list(sk.tags) + ["meta_reflection"],
            )
        )
        result.written_skill_names.append(sk.name)

    for hyp in payload.new_hypotheses:
        append_backlog(hyp, data_dir)
        result.appended_hypotheses.append(hyp)

    return result
