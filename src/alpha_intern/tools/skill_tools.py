"""Tool wrappers around alpha_intern.memory.skills."""

from __future__ import annotations

from pydantic import Field

from alpha_intern.tools.registry import (
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
)


class ListSkillsIn(ToolInput):
    pass


class ListSkillsOut(ToolOutput):
    skills: list[dict]


class GetSkillIn(ToolInput):
    name: str = Field(..., description="Skill name to fetch.")


class GetSkillOut(ToolOutput):
    skill: dict


def _require_skills(ctx: ToolContext) -> None:
    if ctx.skills is None:
        raise ToolError("This tool requires a SkillRegistry in ToolContext")


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        name="list_skills",
        description="List registered research skills (name + description + steps).",
        input_model=ListSkillsIn,
        output_model=ListSkillsOut,
        tags=("skills",),
    )
    def _list(_: ListSkillsIn, ctx: ToolContext) -> ListSkillsOut:
        _require_skills(ctx)
        skills = ctx.skills.list_skills()
        return ListSkillsOut(skills=[s.model_dump() for s in skills])

    @registry.tool(
        name="get_skill",
        description="Fetch a single research skill by name.",
        input_model=GetSkillIn,
        output_model=GetSkillOut,
        tags=("skills",),
    )
    def _get(inp: GetSkillIn, ctx: ToolContext) -> GetSkillOut:
        _require_skills(ctx)
        skill = ctx.skills.get_skill(inp.name)
        return GetSkillOut(skill=skill.model_dump())
