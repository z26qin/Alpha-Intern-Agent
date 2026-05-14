"""Context assembly for the agent loop.

Before the first LLM call we collect:
- the system prompt
- the user's research goal
- the available skill recipes (so the agent can pick one to follow)
- relevant memories (by ticker / tags / keyword query, else most recent)
- the current workspace artifact names (so the agent knows what data
  is already loaded)

Everything is rendered into a single user message body. A coarse
character budget keeps it from running away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from alpha_intern.agent.prompts import SYSTEM_PROMPT
from alpha_intern.memory.skills import SkillRegistry
from alpha_intern.memory.store import ResearchMemoryStore
from alpha_intern.tools.workspace import Workspace


@dataclass
class AssembledContext:
    """Inputs ready to be passed to a `provider.generate(...)` call."""

    system: str
    initial_user_message: str
    skill_names: list[str]
    memory_ids: list[str]
    artifact_names: list[str]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 20)] + "\n... [truncated] ..."


def assemble_context(
    goal: str,
    *,
    memory: Optional[ResearchMemoryStore] = None,
    skills: Optional[SkillRegistry] = None,
    workspace: Optional[Workspace] = None,
    memory_query: Optional[str] = None,
    memory_ticker: Optional[str] = None,
    memory_tags: Optional[list[str]] = None,
    memory_limit: int = 5,
    skill_limit: int = 6,
    memory_content_chars: int = 320,
    char_budget: int = 8000,
    system_prompt: Optional[str] = None,
) -> AssembledContext:
    """Build the first user message for the agent loop."""
    blocks: list[str] = []
    blocks.append("## Research goal")
    blocks.append(goal.strip())

    # Skills
    skill_names: list[str] = []
    if skills is not None:
        available = skills.list_skills()[:skill_limit]
        if available:
            skill_lines = []
            for s in available:
                steps_inline = "; ".join(s.steps[:6])
                skill_lines.append(
                    f"- **{s.name}** — {s.description}\n  steps: {steps_inline}"
                )
                skill_names.append(s.name)
            blocks.append("## Available skill recipes")
            blocks.append("\n".join(skill_lines))

    # Memories
    memory_ids: list[str] = []
    if memory is not None:
        query = memory_query if memory_query is not None else goal
        try:
            relevant = memory.search_memory(
                query=query,
                ticker=memory_ticker,
                tags=memory_tags,
                limit=memory_limit,
            )
        except Exception:
            relevant = []
        if not relevant:
            relevant = memory.list_recent(limit=memory_limit)

        if relevant:
            mem_lines = []
            for m in relevant:
                memory_ids.append(m.id)
                snippet = m.content.replace("\n", " ").strip()
                snippet = _truncate(snippet, memory_content_chars)
                mem_lines.append(
                    f"- [{m.timestamp}] ({m.ticker or '-'}) {m.title}\n  {snippet}"
                )
            blocks.append("## Relevant prior memory notes")
            blocks.append("\n".join(mem_lines))

    # Workspace artifacts
    artifact_names: list[str] = []
    if workspace is not None:
        artifact_names = list(workspace.names())
        if artifact_names:
            blocks.append("## Workspace artifacts already loaded")
            blocks.append(", ".join(artifact_names))

    blocks.append(
        "## Instructions\n"
        "Investigate the goal above. Call tools as needed; you can chain them. "
        "When you have a finding, save a concise summary via `add_memory` "
        "and reply with a short closing message."
    )

    body = "\n\n".join(blocks)
    body = _truncate(body, char_budget)

    return AssembledContext(
        system=system_prompt if system_prompt is not None else SYSTEM_PROMPT,
        initial_user_message=body,
        skill_names=skill_names,
        memory_ids=memory_ids,
        artifact_names=artifact_names,
    )
