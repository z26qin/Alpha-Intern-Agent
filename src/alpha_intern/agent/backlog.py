"""Simple newline-delimited hypothesis backlog.

Lines starting with '#' or blank are ignored. A line beginning with
'[tried=N]' is a retry counter — the research loop bumps it after a
failed attempt and abandons the item after MAX_RETRIES.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAX_RETRIES = 3
_RETRY_RE = re.compile(r"^\[tried=(\d+)\]\s*(.*)$")


@dataclass
class BacklogItem:
    text: str
    tried: int = 0

    def render(self) -> str:
        prefix = f"[tried={self.tried}] " if self.tried else ""
        return prefix + self.text

    @classmethod
    def parse(cls, line: str) -> Optional["BacklogItem"]:
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            return None
        m = _RETRY_RE.match(line.strip())
        if m:
            return cls(text=m.group(2).strip(), tried=int(m.group(1)))
        return cls(text=line.strip(), tried=0)


def _backlog_path(data_dir: Path) -> Path:
    return data_dir / "backlog.txt"


def _done_path(data_dir: Path) -> Path:
    return data_dir / "backlog_done.txt"


def read_backlog(data_dir: Path) -> list[BacklogItem]:
    p = _backlog_path(data_dir)
    if not p.exists():
        return []
    items: list[BacklogItem] = []
    for line in p.read_text().splitlines():
        item = BacklogItem.parse(line)
        if item is not None:
            items.append(item)
    return items


def write_backlog(items: list[BacklogItem], data_dir: Path) -> None:
    p = _backlog_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(it.render() for it in items) + ("\n" if items else ""))


def append_done(text: str, data_dir: Path, status: str = "done") -> None:
    p = _done_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(f"[{status}] {text}\n")


def append_backlog(text: str, data_dir: Path) -> None:
    """Append one new hypothesis (deduped against existing items)."""
    items = read_backlog(data_dir)
    if any(it.text.strip() == text.strip() for it in items):
        return
    items.append(BacklogItem(text=text.strip()))
    write_backlog(items, data_dir)


def pop_next(data_dir: Path) -> Optional[BacklogItem]:
    """Return the first non-exhausted item without removing it."""
    items = read_backlog(data_dir)
    for it in items:
        if it.tried < MAX_RETRIES:
            return it
    return None


def mark_attempt(text: str, data_dir: Path, *, success: bool) -> None:
    """After running an item: on success drop it and log; on failure bump tries."""
    items = read_backlog(data_dir)
    new_items: list[BacklogItem] = []
    matched = False
    for it in items:
        if not matched and it.text.strip() == text.strip():
            matched = True
            if success:
                append_done(it.text, data_dir, status="done")
                continue
            it.tried += 1
            if it.tried >= MAX_RETRIES:
                append_done(it.text, data_dir, status="abandoned")
                continue
        new_items.append(it)
    write_backlog(new_items, data_dir)
