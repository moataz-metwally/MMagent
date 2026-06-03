"""Anthropic Messages API provider for PI-agent."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import httpx
from pi_agent.agent_core import (
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
    Usage,
    UsageCost,
)
from pi_agent.agent_core.event_stream import AssistantMessageEventStream
from pi_agent.agent_core.types import AssistantStream
from pi_agent.pi_ai.types import PiAIRequest

from . import config


class AnthropicProvider:
    """PI-agent Provider that speaks the Anthropic Messages API."""

    def __init__(self, api_key: str = "", base_url: str = ""):
        self._api_key = api_key
        self._base_url = (base_url or "").rstrip("/")

    async def stream(
        self,
        request: PiAIRequest,
        abort_event: asyncio.Event | None = None,
    ) -> AssistantStream:
        stream = AssistantMessageEventStream()
        api_key = request.api_key or self._api_key
        base_url = (request.model.base_url or self._base_url).rstrip("/")
        asyncio.create_task(
            self._emit(stream, request, api_key, base_url, abort_event)
        )
        return stream

    async def _emit(
        self,
        stream: AssistantMessageEventStream,
        request: PiAIRequest,
        api_key: str,
        base_url: str,
        abort_event: asyncio.Event | None,
    ) -> None:
        await asyncio.sleep(0)
        try:
            payload = _build_anthropic_payload(request)
            url = f"{base_url}/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "accept": "text/event-stream",
            }

            partial = _new_partial(request.model)
            stream.push({"type": "start", "partial": partial})

            current_kind: str | None = None
            current_index: int | None = None
            arg_buffer: str = ""

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        error_msg = body.decode(errors="replace")[:500]
                        stream.push({
                            "type": "error",
                            "reason": "error",
                            "error": _error_msg(request.model, error_msg),
                        })
                        return

                    async for line in resp.aiter_lines():
                        if abort_event and abort_event.is_set():
                            stream.push({
                                "type": "error",
                                "reason": "aborted",
                                "error": _error_msg(request.model, "Aborted"),
                            })
                            return

                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break

                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        evt_type = event.get("type", "")

                        if evt_type == "content_block_start":
                            block = (event.get("content_block") or {})
                            btype = block.get("type", "")
                            if btype == "text":
                                current_kind = "text"
                                current_index = len(partial.content)
                                partial.content.append(TextContent(text=""))
                                stream.push({
                                    "type": "text_start",
                                    "content_index": current_index,
                                    "partial": partial,
                                })
                            elif btype == "tool_use":
                                current_kind = "tool"
                                current_index = len(partial.content)
                                arg_buffer = ""
                                tc = ToolCall(
                                    id=block.get("id", ""),
                                    name=block.get("name", ""),
                                    arguments={},
                                )
                                partial.content.append(tc)
                                stream.push({
                                    "type": "toolcall_start",
                                    "content_index": current_index,
                                    "partial": partial,
                                })

                        elif evt_type == "content_block_delta":
                            delta = event.get("delta") or {}
                            dtype = delta.get("type", "")
                            if dtype == "text_delta" and current_kind == "text":
                                text = delta.get("text", "")
                                if text and current_index is not None:
                                    cast_block = partial.content[current_index]
                                    if isinstance(cast_block, TextContent):
                                        cast_block.text += text
                                stream.push({
                                    "type": "text_delta",
                                    "content_index": current_index,
                                    "delta": text,
                                    "partial": partial,
                                })
                            elif dtype == "input_json_delta" and current_kind == "tool":
                                frag = delta.get("partial_json", "")
                                arg_buffer += frag
                                if current_index is not None:
                                    cast_block = partial.content[current_index]
                                    if isinstance(cast_block, ToolCall):
                                        cast_block.arguments = _try_parse_json(arg_buffer)
                                stream.push({
                                    "type": "toolcall_delta",
                                    "content_index": current_index or 0,
                                    "delta": frag,
                                    "partial": partial,
                                })

                        elif evt_type == "content_block_stop":
                            if current_kind == "tool" and current_index is not None:
                                cast_block = partial.content[current_index]
                                if isinstance(cast_block, ToolCall):
                                    cast_block.arguments = _try_parse_json(arg_buffer)
                                stream.push({
                                    "type": "toolcall_end",
                                    "content_index": current_index,
                                    "tool_call": cast_block,
                                    "partial": partial,
                                })
                            elif current_kind == "text" and current_index is not None:
                                cast_block = partial.content[current_index]
                                text_val = cast_block.text if isinstance(cast_block, TextContent) else ""
                                stream.push({
                                    "type": "text_end",
                                    "content_index": current_index,
                                    "content": text_val,
                                    "partial": partial,
                                })
                            current_kind = None
                            current_index = None

                        elif evt_type == "message_delta":
                            mdelta = event.get("delta") or {}
                            stop = mdelta.get("stop_reason")
                            if stop:
                                partial.stop_reason = _map_stop(stop)
                            usage_delta = event.get("usage") or {}
                            if "output_tokens" in usage_delta:
                                partial.usage = Usage(
                                    input=partial.usage.input,
                                    output=usage_delta["output_tokens"],
                                    total_tokens=partial.usage.input + usage_delta["output_tokens"],
                                    cost=UsageCost(),
                                )

                        elif evt_type == "message_start":
                            msg = event.get("message") or {}
                            u = msg.get("usage") or {}
                            partial.usage = Usage(
                                input=u.get("input_tokens", 0),
                                output=u.get("output_tokens", 0),
                                total_tokens=u.get("input_tokens", 0) + u.get("output_tokens", 0),
                                cost=UsageCost(),
                            )

                        elif evt_type == "error":
                            err_data = event.get("error") or {}
                            stream.push({
                                "type": "error",
                                "reason": "error",
                                "error": _error_msg(
                                    request.model,
                                    err_data.get("message", str(event)),
                                ),
                            })
                            return

            if not partial.content:
                partial.content.append(TextContent(text=""))

            reason = "toolUse" if any(isinstance(b, ToolCall) for b in partial.content) else "stop"
            partial.stop_reason = partial.stop_reason or reason

            stream.push({
                "type": "done",
                "reason": reason,
                "message": _clone_assistant(partial),
            })

        except Exception as exc:
            stream.push({
                "type": "error",
                "reason": "error",
                "error": _error_msg(request.model, str(exc)),
            })


# ── Payload helpers ───────────────────────────────────────────────────────────


def _build_anthropic_payload(request: PiAIRequest) -> dict[str, Any]:
    ctx = request.context
    messages: list[dict[str, Any]] = []

    for msg in ctx.messages:
        if hasattr(msg, "role"):
            if msg.role == "user":
                text = msg.content if isinstance(msg.content, str) else " ".join(
                    b.text for b in msg.content if isinstance(b, TextContent)
                )
                messages.append({"role": "user", "content": text})
            elif msg.role == "assistant":
                parts: list[dict[str, Any]] = []
                for b in msg.content:
                    if isinstance(b, TextContent) and b.text:
                        parts.append({"type": "text", "text": b.text})
                    elif isinstance(b, ToolCall):
                        parts.append({
                            "type": "tool_use",
                            "id": b.id,
                            "name": b.name,
                            "input": b.arguments or {},
                        })
                if parts:
                    messages.append({"role": "assistant", "content": parts})
            elif msg.role == "toolResult":
                text = " ".join(
                    b.text for b in msg.content if isinstance(b, TextContent)
                )
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": text,
                    }],
                })

    # Check session-specific override first, fall back to global config
    from .agent import _max_tokens_registry
    _max_tokens = _max_tokens_registry.get(
        request.session_id or "", config.LLM_MAX_TOKENS,
    )

    payload: dict[str, Any] = {
        "model": request.model.id,
        "max_tokens": _max_tokens,
        "messages": messages,
        "stream": True,
    }

    if ctx.system_prompt:
        payload["system"] = ctx.system_prompt

    if ctx.tools:
        anthropic_tools = []
        for tool in ctx.tools:
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": _coerce_schema(tool.parameters),
            })
        payload["tools"] = anthropic_tools

    return payload


def _coerce_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {"type": "object", "properties": {}}
    return {k: v for k, v in schema.items()}


def _new_partial(model: Model) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(cost=UsageCost()),
        stop_reason="stop",
    )


def _clone_assistant(msg: AssistantMessage) -> AssistantMessage:
    content = []
    for b in msg.content:
        if isinstance(b, TextContent):
            content.append(TextContent(text=b.text, text_signature=b.text_signature))
        elif isinstance(b, ToolCall):
            content.append(ToolCall(
                id=b.id, name=b.name, arguments=dict(b.arguments),
                thought_signature=b.thought_signature,
            ))
    return AssistantMessage(
        content=content, api=msg.api, provider=msg.provider,
        model=msg.model, usage=msg.usage, stop_reason=msg.stop_reason,
        error_message=msg.error_message,
    )


def _error_msg(model: Model, text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="")],
        api=model.api, provider=model.provider, model=model.id,
        usage=Usage(cost=UsageCost()), stop_reason="error",
        error_message=text,
    )


def _map_stop(reason: str) -> str:
    if reason in ("end_turn", "stop"):
        return "stop"
    if reason in ("tool_use", "tool_calls"):
        return "toolUse"
    if reason == "max_tokens":
        return "length"
    return "stop"


def _try_parse_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
