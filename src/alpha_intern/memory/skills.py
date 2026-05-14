"""Deterministic research skill registry.

A "skill" is a named, declarative recipe an analyst (human or LLM) can
follow. It is intentionally not executable code: skills describe *what*
to do, while modules under `alpha_intern/` provide the *how*.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchSkill(BaseModel):
    """A named research recipe."""

    name: str
    description: str
    inputs: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


def _default_skills() -> list[ResearchSkill]:
    return [
        ResearchSkill(
            name="momentum_signal_research",
            description=(
                "Investigate cross-sectional price momentum as a return"
                " predictor over a chosen universe and horizon."
            ),
            inputs=["price_panel", "universe", "lookback_days", "holding_days"],
            steps=[
                "Normalize price data via data.loader.normalize_price_dataframe.",
                "Build returns and momentum features via features.technical.",
                "Rank stocks by momentum on each date.",
                "Run long/short rank backtest with run_simple_rank_backtest.",
                "Record findings (Sharpe, drawdown, caveats) as a memory note.",
            ],
            outputs=["backtest_returns", "summary_metrics", "memory_note"],
            tags=["momentum", "cross_sectional", "signal"],
        ),
        ResearchSkill(
            name="earnings_reaction_backtest",
            description=(
                "Study post-earnings drift / reversal in a universe by"
                " grouping returns relative to earnings announcement dates."
            ),
            inputs=["price_panel", "earnings_dates", "event_window"],
            steps=[
                "Align price returns to event time (t-N ... t+N).",
                "Compute average abnormal return per event-time bucket.",
                "Test whether drift is statistically/economically meaningful.",
                "Save the event-study table as a memory note.",
            ],
            outputs=["event_study_table", "memory_note"],
            tags=["events", "earnings", "drift"],
        ),
        ResearchSkill(
            name="factor_decay_analysis",
            description=(
                "Measure how quickly a signal's predictive power decays"
                " across forward-return horizons (1d, 5d, 20d, 60d)."
            ),
            inputs=["signal_panel", "forward_returns_panel"],
            steps=[
                "Compute IC (rank correlation) of signal vs. forward returns at each horizon.",
                "Plot or tabulate IC by horizon.",
                "Identify the horizon where IC is highest and where it decays to zero.",
                "Note implications for rebalance frequency in a memory note.",
            ],
            outputs=["ic_by_horizon", "memory_note"],
            tags=["factor", "ic", "horizon"],
        ),
        ResearchSkill(
            name="transaction_cost_sensitivity",
            description=(
                "Sweep transaction-cost assumptions to find the cost level"
                " at which a strategy stops being profitable."
            ),
            inputs=["backtest_inputs", "cost_bps_grid"],
            steps=[
                "Re-run run_simple_rank_backtest across a grid of cost_bps values.",
                "Compute net Sharpe and max drawdown at each cost level.",
                "Identify the break-even cost.",
                "Record break-even cost and turnover caveats in a memory note.",
            ],
            outputs=["cost_sensitivity_table", "memory_note"],
            tags=["costs", "sensitivity", "robustness"],
        ),
    ]


class SkillRegistry:
    """JSON-backed registry of research skills."""

    def __init__(
        self,
        path: str | Path,
        seed_defaults: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, ResearchSkill] = {}

        if self.path.exists():
            self._load()
        elif seed_defaults:
            for skill in _default_skills():
                self._skills[skill.name] = skill
            self._save()

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        self._skills = {
            name: ResearchSkill.model_validate(payload)
            for name, payload in raw.items()
        }

    def _save(self) -> None:
        payload = {
            name: json.loads(skill.model_dump_json())
            for name, skill in self._skills.items()
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def add_skill(self, skill: ResearchSkill) -> ResearchSkill:
        if skill.name in self._skills:
            raise ValueError(f"Skill {skill.name!r} already exists")
        self._skills[skill.name] = skill
        self._save()
        return skill

    def get_skill(self, name: str) -> ResearchSkill:
        if name not in self._skills:
            raise KeyError(f"No such skill: {name!r}")
        return self._skills[name]

    def list_skills(self) -> list[ResearchSkill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def update_skill(self, name: str, **updates) -> ResearchSkill:
        if name not in self._skills:
            raise KeyError(f"No such skill: {name!r}")
        existing = self._skills[name]
        data = existing.model_dump()
        for key, value in updates.items():
            if key not in data:
                raise ValueError(f"Unknown skill field: {key!r}")
            data[key] = value
        data["updated_at"] = _now_iso()
        updated = ResearchSkill.model_validate(data)
        self._skills[name] = updated
        self._save()
        return updated

    def remove_skill(self, name: str) -> None:
        if name not in self._skills:
            raise KeyError(f"No such skill: {name!r}")
        del self._skills[name]
        self._save()
