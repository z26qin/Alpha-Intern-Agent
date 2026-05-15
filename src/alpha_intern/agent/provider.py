"""LLM provider abstraction.

Two implementations:

- `AnthropicProvider` — wraps the official Anthropic Python SDK. Requires
  `pip install alpha-intern-agent[agent]` and `ANTHROPIC_API_KEY` in env.
- `ScriptedProvider` — returns a queued sequence of `LLMResponse`s.
  Used by tests so the loop can be exercised without network access.

Keeping the surface tiny: provider.generate(system, messages, tools,
max_tokens) -> LLMResponse. Anything richer (streaming, model routing,
thinking) is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass
class ToolUse:
    """A tool_use block from the model's response."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Provider-neutral response."""

    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can take (system, messages, tools) and return a response."""

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> LLMResponse: ...


class AnthropicProvider:
    """Real provider that talks to the Anthropic API.

    The SDK is imported lazily so the package remains importable without
    the optional `[agent]` extra installed.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - environmental
            raise ImportError(
                "The Anthropic SDK is not installed. Install it with "
                "`pip install alpha-intern-agent[agent]`."
            ) from exc

        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.model = model

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> LLMResponse:  # pragma: no cover - requires network + API key
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )

        text_parts: list[str] = []
        tool_uses: list[ToolUse] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_uses.append(
                    ToolUse(
                        id=getattr(block, "id"),
                        name=getattr(block, "name"),
                        input=dict(getattr(block, "input", {}) or {}),
                    )
                )

        usage: dict[str, int] = {}
        raw_usage = getattr(resp, "usage", None)
        if raw_usage is not None:
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                val = getattr(raw_usage, key, None)
                if isinstance(val, int):
                    usage[key] = val

        return LLMResponse(
            text="\n".join(p for p in text_parts if p),
            tool_uses=tool_uses,
            stop_reason=getattr(resp, "stop_reason", None) or "end_turn",
            raw=resp,
            model=getattr(resp, "model", self.model) or self.model,
            usage=usage,
        )


class ScriptedProvider:
    """Test-only provider that returns a queued list of responses.

    Records every call so tests can assert what the loop sent.
    """

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self._responses: list[LLMResponse] = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise RuntimeError("ScriptedProvider has no more responses queued")
        return self._responses.pop(0)
