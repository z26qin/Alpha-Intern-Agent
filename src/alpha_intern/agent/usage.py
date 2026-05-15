"""Read run logs and summarize Claude API usage / cost per run.

Pricing is a small hardcoded table keyed by model-name prefix. Costs
are estimates and may lag actual Anthropic pricing; treat them as
order-of-magnitude indicators, not invoices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from alpha_intern.agent.run_log import RunLog, RunLogEntry


# (input $/MTok, output $/MTok). Cache write = 1.25x input, cache read = 0.1x input.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (1.0, 5.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-opus": (15.0, 75.0),
}


def _price_for(model: str) -> Optional[tuple[float, float]]:
    if not model:
        return None
    for prefix, rates in _PRICING.items():
        if model.startswith(prefix):
            return rates
    return None


@dataclass
class RunUsage:
    run_id: str
    started: str
    model: str
    n_calls: int
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    duration_s: float

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )

    def estimated_cost_usd(self) -> Optional[float]:
        rates = _price_for(self.model)
        if rates is None:
            return None
        in_rate, out_rate = rates
        cost = (
            self.input_tokens * in_rate
            + self.output_tokens * out_rate
            + self.cache_creation_tokens * in_rate * 1.25
            + self.cache_read_tokens * in_rate * 0.10
        ) / 1_000_000
        return cost


def summarize_runs(path: str | Path) -> list[RunUsage]:
    """Walk the JSONL log and collapse llm_call entries into per-run rows."""
    p = Path(path)
    if not p.exists():
        return []
    log = RunLog(p, write_start_event=False)
    bucket: dict[str, RunUsage] = {}
    for entry in log.iter_entries():
        if entry.step_type == "run_start":
            bucket.setdefault(
                entry.run_id,
                RunUsage(
                    run_id=entry.run_id,
                    started=entry.timestamp,
                    model="",
                    n_calls=0,
                    input_tokens=0,
                    output_tokens=0,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    duration_s=0.0,
                ),
            )
        elif entry.step_type == "llm_call":
            row = bucket.setdefault(
                entry.run_id,
                RunUsage(
                    run_id=entry.run_id,
                    started=entry.timestamp,
                    model="",
                    n_calls=0,
                    input_tokens=0,
                    output_tokens=0,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    duration_s=0.0,
                ),
            )
            usage = entry.metadata.get("usage") or {}
            row.n_calls += 1
            row.input_tokens += int(usage.get("input_tokens") or 0)
            row.output_tokens += int(usage.get("output_tokens") or 0)
            row.cache_creation_tokens += int(usage.get("cache_creation_input_tokens") or 0)
            row.cache_read_tokens += int(usage.get("cache_read_input_tokens") or 0)
            row.duration_s += float(entry.duration_s or 0.0)
            model = entry.metadata.get("model") or ""
            if model and not row.model:
                row.model = model
    return list(bucket.values())


def render_table(rows: Iterable[RunUsage]) -> str:
    rows = list(rows)
    if not rows:
        return "(no LLM calls recorded yet)"

    header = (
        f"{'run_id':<14}{'started':<22}{'model':<22}"
        f"{'calls':>6}{'in':>10}{'out':>10}{'c_w':>8}{'c_r':>8}"
        f"{'sec':>7}{'cost$':>9}"
    )
    lines = [header, "-" * len(header)]

    tot_calls = tot_in = tot_out = tot_cw = tot_cr = 0
    tot_sec = 0.0
    tot_cost = 0.0
    known_cost = True

    for r in rows:
        cost = r.estimated_cost_usd()
        cost_str = f"{cost:>9.4f}" if cost is not None else f"{'?':>9}"
        if cost is None:
            known_cost = False
        else:
            tot_cost += cost
        lines.append(
            f"{r.run_id:<14}{r.started[:19]:<22}{(r.model or '?'):<22}"
            f"{r.n_calls:>6}{r.input_tokens:>10}{r.output_tokens:>10}"
            f"{r.cache_creation_tokens:>8}{r.cache_read_tokens:>8}"
            f"{r.duration_s:>7.1f}{cost_str}"
        )
        tot_calls += r.n_calls
        tot_in += r.input_tokens
        tot_out += r.output_tokens
        tot_cw += r.cache_creation_tokens
        tot_cr += r.cache_read_tokens
        tot_sec += r.duration_s

    lines.append("-" * len(header))
    cost_total_str = f"{tot_cost:>9.4f}" if known_cost else f"{'?':>9}"
    lines.append(
        f"{'TOTAL':<14}{'':<22}{'':<22}"
        f"{tot_calls:>6}{tot_in:>10}{tot_out:>10}{tot_cw:>8}{tot_cr:>8}"
        f"{tot_sec:>7.1f}{cost_total_str}"
    )
    return "\n".join(lines)
