"""Per-session structured run logs.

Each `RunLog` is a JSONL file (one entry per line) recording the steps
of a single research session: tool calls, results, errors, and free-form
events ("plan", "observation", "note"). This is the trace a future
self-improvement loop will read.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Optional

from pydantic import BaseModel, Field

StepType = Literal[
    "run_start",
    "run_end",
    "run_end_summary",
    "tool_call",
    "tool_error",
    "plan",
    "provider_error",
    "observation",
    "note",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class RunLogEntry(BaseModel):
    """A single line in a run log."""

    timestamp: str = Field(default_factory=_now_iso)
    run_id: str
    step_type: StepType
    tool_name: Optional[str] = None
    input: Optional[dict[str, Any]] = None
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    duration_s: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunLog:
    """JSONL-backed run log for a single research session."""

    def __init__(
        self,
        path: str | Path,
        run_id: Optional[str] = None,
        write_start_event: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self.run_id = run_id or _new_run_id()
        self._closed = False
        if write_start_event:
            self.log_event("run_start")

    def append(self, entry: RunLogEntry) -> RunLogEntry:
        if self._closed:
            raise RuntimeError("Cannot append to a closed RunLog")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
        return entry

    def log_event(
        self,
        step_type: StepType,
        **metadata: Any,
    ) -> RunLogEntry:
        entry = RunLogEntry(
            run_id=self.run_id,
            step_type=step_type,
            metadata=metadata,
        )
        return self.append(entry)

    def log_tool_call(
        self,
        tool_name: str,
        input: dict[str, Any],
        output: dict[str, Any],
        duration_s: Optional[float] = None,
        **metadata: Any,
    ) -> RunLogEntry:
        entry = RunLogEntry(
            run_id=self.run_id,
            step_type="tool_call",
            tool_name=tool_name,
            input=_safe_dict(input),
            output=_safe_dict(output),
            duration_s=duration_s,
            metadata=metadata,
        )
        return self.append(entry)

    def log_tool_error(
        self,
        tool_name: str,
        input: dict[str, Any],
        error: str,
        duration_s: Optional[float] = None,
        **metadata: Any,
    ) -> RunLogEntry:
        entry = RunLogEntry(
            run_id=self.run_id,
            step_type="tool_error",
            tool_name=tool_name,
            input=_safe_dict(input),
            error=error,
            duration_s=duration_s,
            metadata=metadata,
        )
        return self.append(entry)

    def log_note(self, content: str, **metadata: Any) -> RunLogEntry:
        return self.append(
            RunLogEntry(
                run_id=self.run_id,
                step_type="note",
                metadata={"content": content, **metadata},
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self.log_event("run_end")
        self._closed = True

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def iter_entries(self, run_id: Optional[str] = None) -> Iterator[RunLogEntry]:
        target = run_id if run_id is not None else None
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = RunLogEntry.model_validate_json(line)
                if target is None or entry.run_id == target:
                    yield entry

    def entries(self, run_id: Optional[str] = None) -> list[RunLogEntry]:
        return list(self.iter_entries(run_id=run_id))

    def current_run_entries(self) -> list[RunLogEntry]:
        return self.entries(run_id=self.run_id)


def _safe_dict(d: Any) -> dict[str, Any]:
    """Best-effort coerce a value to a JSON-serializable dict for logging."""
    if d is None:
        return {}
    if isinstance(d, dict):
        try:
            json.dumps(d, default=str)
            return d
        except (TypeError, ValueError):
            return {k: _safe(v) for k, v in d.items()}
    return {"value": _safe(d)}


def _safe(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


def iter_run_ids(path: str | Path) -> Iterable[str]:
    """Yield distinct run_ids present in a JSONL log file, in order."""
    seen: set[str] = set()
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = RunLogEntry.model_validate_json(line)
            if entry.run_id not in seen:
                seen.add(entry.run_id)
                yield entry.run_id
