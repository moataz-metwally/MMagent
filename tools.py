"""Tavily tool definitions and JSON schemas for PI-agent."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any

from pi_agent.agent_core import AgentToolResult, TextContent
from tavily import TavilyClient

from . import config
from .config import DepthProfile

# Active depth profile — set via set_profile() before tools are called
_profile: DepthProfile = config.DEPTH_PROFILES["standard"]

# Module-level Tavily key — set via set_tavily_key() or falls back to config
_tavily_key: str = config.TAVILY_API_KEY


def set_profile(profile: DepthProfile) -> None:
    global _profile
    _profile = profile


def set_tavily_key(key: str) -> None:
    global _tavily_key
    _tavily_key = key


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_tavily() -> TavilyClient:
    return TavilyClient(api_key=_tavily_key)


def _text_result(text: str) -> AgentToolResult[Any]:
    return AgentToolResult(content=[TextContent(text=text)], details=None)


def _json_result(data: Any) -> AgentToolResult[Any]:
    return _text_result(json.dumps(data, indent=2, ensure_ascii=False))


# ── Tool implementations ─────────────────────────────────────────────────────


async def tavily_search(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[Any], None] | None = None,
) -> AgentToolResult[Any]:
    client = _get_tavily()
    result = client.search(
        query=params["query"],
        max_results=params.get("max_results", _profile.max_results),
        search_depth=params.get("search_depth", _profile.search_depth),
        topic=params.get("topic", "general"),
        include_answer=True,
        include_raw_content=params.get("include_raw_content", _profile.include_raw_content),
        include_domains=params.get("include_domains"),
        exclude_domains=params.get("exclude_domains"),
    )
    return _json_result(result)


async def tavily_extract(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[Any], None] | None = None,
) -> AgentToolResult[Any]:
    client = _get_tavily()
    urls = params["urls"]
    if isinstance(urls, str):
        urls = [urls]
    result = client.extract(
        urls=urls,
        extract_depth=params.get("extract_depth", _profile.extract_depth),
        format=params.get("format", "markdown"),
        query=params.get("query"),
    )
    return _json_result(result)


async def tavily_crawl(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[Any], None] | None = None,
) -> AgentToolResult[Any]:
    client = _get_tavily()
    result = client.crawl(
        url=params["url"],
        max_depth=params.get("max_depth", _profile.crawl_max_depth),
        max_breadth=params.get("max_breadth", _profile.crawl_max_breadth),
        limit=params.get("limit", _profile.crawl_limit),
        instructions=params.get("instructions"),
        select_paths=params.get("select_paths"),
        select_domains=params.get("select_domains"),
        extract_depth=params.get("extract_depth", _profile.extract_depth),
        format=params.get("format", "markdown"),
    )
    return _json_result(result)


async def tavily_map(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[Any], None] | None = None,
) -> AgentToolResult[Any]:
    client = _get_tavily()
    result = client.map(
        url=params["url"],
        max_depth=params.get("max_depth", _profile.map_max_depth),
        max_breadth=params.get("max_breadth", _profile.crawl_max_breadth),
        limit=params.get("limit", _profile.map_limit),
        instructions=params.get("instructions"),
        select_paths=params.get("select_paths"),
        select_domains=params.get("select_domains"),
    )
    return _json_result(result)


async def tavily_research(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[Any], None] | None = None,
) -> AgentToolResult[Any]:
    client = _get_tavily()
    resp = client.research(
        input=params["input"],
        model=params.get("model", _profile.research_model),
    )
    request_id = resp.get("request_id")
    if not request_id:
        return _json_result(resp)

    for _ in range(60):
        if abort_event and abort_event.is_set():
            return _text_result("Research aborted.")
        result = client.get_research(request_id)
        status = result.get("status", "")
        if status in ("complete", "completed", "done"):
            return _json_result(result)
        if status in ("failed", "error"):
            return _json_result(result)
        await asyncio.sleep(3)

    return _text_result("Research timed out after ~3 minutes.")


async def tavily_dynamic_search(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[Any], None] | None = None,
) -> AgentToolResult[Any]:
    client = _get_tavily()
    result = client.search(
        query=params["query"],
        max_results=params.get("max_results", _profile.max_results),
        search_depth=params.get("search_depth", _profile.search_depth),
        topic=params.get("topic", "general"),
        include_raw_content=params.get("include_raw_content", _profile.include_raw_content),
        include_answer=params.get("include_answer", True),
        include_domains=params.get("include_domains"),
        exclude_domains=params.get("exclude_domains"),
    )
    return _json_result(result)


# ── JSON schemas ─────────────────────────────────────────────────────────────


SEARCH_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query"},
        "max_results": {"type": "integer", "description": "Max results to return"},
        "search_depth": {
            "type": "string",
            "enum": ["basic", "advanced", "fast", "ultra-fast"],
        },
        "topic": {
            "type": "string",
            "enum": ["general", "news", "finance"],
        },
        "include_domains": {
            "type": "array",
            "items": {"type": "string"},
        },
        "exclude_domains": {
            "type": "array",
            "items": {"type": "string"},
        },
        "include_raw_content": {"type": "boolean"},
    },
    "required": ["query"],
}

EXTRACT_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "urls": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ],
            "description": "URL(s) to extract content from",
        },
        "extract_depth": {"type": "string", "enum": ["basic", "advanced"]},
        "query": {"type": "string", "description": "Optional query to focus extraction"},
        "format": {"type": "string", "enum": ["markdown", "text"]},
    },
    "required": ["urls"],
}

CRAWL_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "Root URL to crawl"},
        "max_depth": {"type": "integer"},
        "max_breadth": {"type": "integer"},
        "limit": {"type": "integer"},
        "instructions": {"type": "string"},
        "select_paths": {"type": "array", "items": {"type": "string"}},
        "extract_depth": {"type": "string", "enum": ["basic", "advanced"]},
    },
    "required": ["url"],
}

MAP_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "Root URL to map"},
        "max_depth": {"type": "integer"},
        "limit": {"type": "integer"},
        "instructions": {"type": "string"},
        "select_paths": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["url"],
}

RESEARCH_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "input": {"type": "string", "description": "Research question or topic"},
        "model": {"type": "string", "enum": ["mini", "pro", "auto"]},
    },
    "required": ["input"],
}

DYNAMIC_SEARCH_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "max_results": {"type": "integer"},
        "search_depth": {"type": "string", "enum": ["basic", "advanced", "fast", "ultra-fast"]},
        "topic": {"type": "string", "enum": ["general", "news", "finance"]},
        "include_domains": {"type": "array", "items": {"type": "string"}},
        "exclude_domains": {"type": "array", "items": {"type": "string"}},
        "include_raw_content": {"type": "boolean"},
        "include_answer": {"type": "boolean"},
    },
    "required": ["query"],
}

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "tavily_search": SEARCH_PARAMS,
    "tavily_extract": EXTRACT_PARAMS,
    "tavily_crawl": CRAWL_PARAMS,
    "tavily_map": MAP_PARAMS,
    "tavily_research": RESEARCH_PARAMS,
    "tavily_dynamic_search": DYNAMIC_SEARCH_PARAMS,
}
