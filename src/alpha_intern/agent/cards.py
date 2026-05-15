"""Compact per-run summaries ("cards") that the meta-reflector reads.

A card is one JSON file under data_dir/cards/<run_id>.json. It captures
just enough to spot patterns across runs without re-parsing the raw
JSONL trace each time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class RunCard:
    run_id: str
    created_at: str
    goal: str
    final_text: str = ""
    stopped_reason: str = ""
    steps_used: int = 0
    tools_used: list[str] = field(default_factory=list)
    error_count: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)
    notes: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pluck_metrics(tool_calls: list[dict[str, Any]]) -> dict[str, float]:
    """Find the last compute_metrics output, if any."""
    metric_keys = {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    }
    out: dict[str, float] = {}
    for tc in tool_calls:
        if tc.get("tool") == "compute_metrics" and tc.get("ok") and tc.get("output"):
            for k, v in tc["output"].items():
                if k in metric_keys and isinstance(v, (int, float)):
                    out[k] = float(v)
    return out


def build_card(
    *,
    run_id: str,
    goal: str,
    final_text: str,
    stopped_reason: str,
    steps_used: int,
    tool_calls: list[dict[str, Any]],
    lessons: Optional[list[str]] = None,
    final_text_max: int = 600,
) -> RunCard:
    tools_used = [tc.get("tool", "?") for tc in tool_calls]
    errors = sum(1 for tc in tool_calls if not tc.get("ok"))
    metrics = _pluck_metrics(tool_calls)
    snippet = (final_text or "").strip()
    if len(snippet) > final_text_max:
        snippet = snippet[: final_text_max - 3] + "..."
    return RunCard(
        run_id=run_id,
        created_at=_now_iso(),
        goal=goal,
        final_text=snippet,
        stopped_reason=stopped_reason,
        steps_used=steps_used,
        tools_used=tools_used,
        error_count=errors,
        metrics=metrics,
        lessons=lessons or [],
    )


def write_card(card: RunCard, data_dir: Path) -> Path:
    out_dir = data_dir / "cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{card.run_id}.json"
    path.write_text(json.dumps(asdict(card), indent=2))
    return path


def read_cards(data_dir: Path, limit: Optional[int] = None) -> list[RunCard]:
    out_dir = data_dir / "cards"
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if limit is not None:
        files = files[-limit:]
    cards: list[RunCard] = []
    for p in files:
        try:
            data = json.loads(p.read_text())
            cards.append(RunCard(**data))
        except Exception:
            continue
    return cards
