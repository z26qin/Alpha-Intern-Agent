"""Executor for `ResearchSkill.executable_steps`.

A `SkillRunner` walks a skill's executable steps, resolves
``$param.<name>`` / ``$step.<alias>.<field>`` references against the
caller's params + previous outputs, and dispatches each step through
a `ToolRegistry`. The first failure aborts the run and is reported in
the result; the tool registry's run-log wiring still captures every
individual tool call.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from alpha_intern.memory.skills import ExecutableStep, ResearchSkill
from alpha_intern.tools.registry import ToolContext, ToolRegistry


class SkillRunError(RuntimeError):
    """Raised when a skill cannot be executed (bad refs, missing tool, etc.)."""


class SkillStepResult(BaseModel):
    step_index: int
    tool: str
    inputs: dict[str, Any]
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    ok: bool = False
    alias: Optional[str] = None


class SkillRunResult(BaseModel):
    skill_name: str
    ok: bool
    completed_steps: int
    total_steps: int
    step_results: list[SkillStepResult] = Field(default_factory=list)
    error: Optional[str] = None
    params_used: dict[str, Any] = Field(default_factory=dict)


def _resolve_value(
    value: Any,
    params: dict[str, Any],
    step_outputs: dict[str, dict[str, Any]],
) -> Any:
    if isinstance(value, str):
        if value.startswith("$param."):
            key = value[len("$param.") :]
            if key not in params:
                raise SkillRunError(f"Missing required param: {key!r}")
            return params[key]
        if value.startswith("$step."):
            parts = value[len("$step.") :].split(".", 1)
            if len(parts) != 2:
                raise SkillRunError(
                    f"Bad $step reference {value!r}; expected $step.<alias>.<field>"
                )
            alias, field = parts
            if alias not in step_outputs:
                raise SkillRunError(f"Unknown step alias: {alias!r}")
            if field not in step_outputs[alias]:
                raise SkillRunError(
                    f"Step alias {alias!r} has no field {field!r}; "
                    f"available: {sorted(step_outputs[alias])}"
                )
            return step_outputs[alias][field]
        return value
    if isinstance(value, list):
        return [_resolve_value(v, params, step_outputs) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_value(v, params, step_outputs) for k, v in value.items()}
    return value


def _resolve_step_inputs(
    step: ExecutableStep,
    params: dict[str, Any],
    step_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {k: _resolve_value(v, params, step_outputs) for k, v in step.inputs.items()}


class SkillRunner:
    """Executes a `ResearchSkill`'s executable_steps via a tool registry."""

    def __init__(self, registry: ToolRegistry, ctx: ToolContext) -> None:
        self.registry = registry
        self.ctx = ctx

    def run(
        self,
        skill: Union[str, ResearchSkill],
        *,
        params: Optional[dict[str, Any]] = None,
    ) -> SkillRunResult:
        if isinstance(skill, str):
            if self.ctx.skills is None:
                raise SkillRunError(
                    "Cannot resolve skill by name without a SkillRegistry in ToolContext"
                )
            skill_obj = self.ctx.skills.get_skill(skill)
        else:
            skill_obj = skill

        if not skill_obj.executable_steps:
            return SkillRunResult(
                skill_name=skill_obj.name,
                ok=False,
                completed_steps=0,
                total_steps=0,
                error="Skill has no executable_steps",
            )

        merged_params: dict[str, Any] = dict(skill_obj.default_params)
        if params:
            merged_params.update(params)

        if self.ctx.run_log is not None:
            self.ctx.run_log.log_event(
                "plan",
                source="skill_runner",
                skill_name=skill_obj.name,
                params=_safe_for_log(merged_params),
                n_steps=len(skill_obj.executable_steps),
            )

        step_outputs: dict[str, dict[str, Any]] = {}
        results: list[SkillStepResult] = []

        for idx, step in enumerate(skill_obj.executable_steps):
            try:
                resolved = _resolve_step_inputs(step, merged_params, step_outputs)
            except SkillRunError as exc:
                results.append(
                    SkillStepResult(
                        step_index=idx,
                        tool=step.tool,
                        inputs=step.inputs,
                        error=str(exc),
                        ok=False,
                        alias=step.save_outputs_as,
                    )
                )
                return SkillRunResult(
                    skill_name=skill_obj.name,
                    ok=False,
                    completed_steps=idx,
                    total_steps=len(skill_obj.executable_steps),
                    step_results=results,
                    error=f"step {idx} ({step.tool}): {exc}",
                    params_used=merged_params,
                )

            try:
                output = self.registry.dispatch(
                    step.tool, inputs=resolved, ctx=self.ctx
                )
                payload = output.model_dump()
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                results.append(
                    SkillStepResult(
                        step_index=idx,
                        tool=step.tool,
                        inputs=resolved,
                        error=err,
                        ok=False,
                        alias=step.save_outputs_as,
                    )
                )
                return SkillRunResult(
                    skill_name=skill_obj.name,
                    ok=False,
                    completed_steps=idx,
                    total_steps=len(skill_obj.executable_steps),
                    step_results=results,
                    error=f"step {idx} ({step.tool}): {err}",
                    params_used=merged_params,
                )

            if step.save_outputs_as:
                step_outputs[step.save_outputs_as] = payload

            results.append(
                SkillStepResult(
                    step_index=idx,
                    tool=step.tool,
                    inputs=resolved,
                    output=payload,
                    ok=True,
                    alias=step.save_outputs_as,
                )
            )

        return SkillRunResult(
            skill_name=skill_obj.name,
            ok=True,
            completed_steps=len(results),
            total_steps=len(skill_obj.executable_steps),
            step_results=results,
            params_used=merged_params,
        )


def _safe_for_log(d: dict[str, Any]) -> dict[str, Any]:
    import json

    try:
        json.dumps(d, default=str)
        return d
    except (TypeError, ValueError):
        return {k: str(v) for k, v in d.items()}
