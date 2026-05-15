"""Tool wrappers around alpha_intern.memory.skills."""

from __future__ import annotations

from typing import Any, Optional

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


class RunSkillIn(ToolInput):
    skill_name: str = Field(..., description="Name of a skill with executable_steps.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Overrides for the skill's default_params.",
    )


class RunSkillOut(ToolOutput):
    skill_name: str
    ok: bool
    completed_steps: int
    total_steps: int
    error: Optional[str] = None
    step_results: list[dict] = Field(default_factory=list)


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

    @registry.tool(
        name="run_skill",
        description=(
            "Execute a registered skill's executable_steps end-to-end. "
            "Each step is dispatched as a tool call with `$param.` / "
            "`$step.<alias>.<field>` references resolved against the "
            "provided params and prior step outputs."
        ),
        input_model=RunSkillIn,
        output_model=RunSkillOut,
        tags=("skills",),
    )
    def _run(inp: RunSkillIn, ctx: ToolContext) -> RunSkillOut:
        _require_skills(ctx)
        from alpha_intern.memory.skill_runner import SkillRunner

        runner = SkillRunner(registry=registry, ctx=ctx)
        result = runner.run(inp.skill_name, params=inp.params)
        return RunSkillOut(
            skill_name=result.skill_name,
            ok=result.ok,
            completed_steps=result.completed_steps,
            total_steps=result.total_steps,
            error=result.error,
            step_results=[s.model_dump() for s in result.step_results],
        )
