"""Agent factory, convenience functions, and event handling."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from pi_agent.agent_core import (
    Agent,
    AgentEvent,
    AgentTool,
    AssistantMessage,
    Model,
    TextContent,
)
from pi_agent.pi_ai import create_agent_stream_fn
from pi_agent.pi_ai.registry import ProviderRegistry

from . import config
from .provider import AnthropicProvider
from .tools import (
    DYNAMIC_SEARCH_PARAMS,
    CRAWL_PARAMS,
    EXTRACT_PARAMS,
    MAP_PARAMS,
    RESEARCH_PARAMS,
    SEARCH_PARAMS,
    set_profile,
    set_tavily_key,
    tavily_crawl,
    tavily_dynamic_search,
    tavily_extract,
    tavily_map,
    tavily_research,
    tavily_search,
)

# ── Console colors ───────────────────────────────────────────────────────────

TOOL_COLORS = {
    "tavily_search": "\033[94m",
    "tavily_extract": "\033[95m",
    "tavily_crawl": "\033[93m",
    "tavily_map": "\033[96m",
    "tavily_research": "\033[91m",
    "tavily_dynamic_search": "\033[92m",
}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


# ── Default console event handler ────────────────────────────────────────────


def on_event(event: AgentEvent) -> None:
    etype = event["type"]

    if etype == "text_delta":
        delta = event.get("delta", "")
        print(delta, end="", flush=True)

    elif etype == "toolcall_end":
        tc = event.get("tool_call")
        if tc:
            color = TOOL_COLORS.get(tc.name, "")
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            if len(args_str) > 120:
                args_str = args_str[:117] + "..."
            print(f"\n{color}{BOLD}[{tc.name}]{RESET} {DIM}{args_str}{RESET}")

    elif etype == "tool_execution_end":
        msg = event.get("message")
        if msg and hasattr(msg, "content"):
            for block in msg.content:
                if isinstance(block, TextContent) and block.text:
                    text = block.text
                    if len(text) > 800:
                        text = text[:797] + "..."
                    print(f"  {DIM}{text}{RESET}\n")

    elif etype == "done":
        reason = event.get("reason", "")
        if reason == "toolUse":
            print()


# ── MMAgent wrapper ───────────────────────────────────────────────────────────


# session_id → max_tokens override, read by provider._build_anthropic_payload
_max_tokens_registry: dict[str, int] = {}


class MMAgent:
    """Thin wrapper around PI-agent's Agent that returns the answer from prompt()."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    async def prompt(self, input_value: str, **kwargs: Any) -> str:
        """Send a prompt and return the final answer text."""
        await self._agent.prompt(input_value, **kwargs)
        return extract_final_text(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


def extract_final_text(agent: MMAgent | Agent) -> str:
    state = agent.state if isinstance(agent, MMAgent) else agent.state
    for msg in reversed(state.messages):
        if isinstance(msg, AssistantMessage):
            return " ".join(
                block.text for block in msg.content if isinstance(block, TextContent)
            ).strip()
    return ""


def build_all_tools() -> list[AgentTool]:
    return [
        AgentTool(
            name="tavily_search",
            label="Web Search",
            description="Search the web for current information on any topic.",
            execute=tavily_search,
            parameters=SEARCH_PARAMS,
        ),
        AgentTool(
            name="tavily_extract",
            label="URL Extract",
            description="Extract clean markdown or text content from one or more URLs.",
            execute=tavily_extract,
            parameters=EXTRACT_PARAMS,
        ),
        AgentTool(
            name="tavily_crawl",
            label="Site Crawl",
            description="Crawl a website and extract content from multiple pages.",
            execute=tavily_crawl,
            parameters=CRAWL_PARAMS,
        ),
        AgentTool(
            name="tavily_map",
            label="Site Map",
            description="Discover and list all URLs on a website without extracting content.",
            execute=tavily_map,
            parameters=MAP_PARAMS,
        ),
        AgentTool(
            name="tavily_research",
            label="Deep Research",
            description="Conduct comprehensive multi-source AI-powered research with citations.",
            execute=tavily_research,
            parameters=RESEARCH_PARAMS,
        ),
        AgentTool(
            name="tavily_dynamic_search",
            label="Dynamic Search",
            description="Programmatic search with advanced filtering for noise-free curated results.",
            execute=tavily_dynamic_search,
            parameters=DYNAMIC_SEARCH_PARAMS,
        ),
    ]


# ── Agent factory ─────────────────────────────────────────────────────────────


def create_agent(
    *,
    depth: str = "standard",
    llm_model_id: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
    llm_provider: str | None = None,
    llm_api_type: str | None = None,
    tavily_api_key: str | None = None,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    on_event: Callable[[AgentEvent], None] | None = None,
    extra_tools: list[AgentTool] | None = None,
    disable_search: bool = False,
) -> MMAgent:
    """Create a configured agent.

    All parameters default to values from config.py. Pass non-None values to override.
    When disable_search=True, no Tavily search tools are loaded — only extra_tools.
    Returns an MMAgent whose prompt() method returns the answer directly.
    """
    profile = config.DEPTH_PROFILES[depth]
    set_profile(profile)

    if tavily_api_key is not None:
        set_tavily_key(tavily_api_key)

    _api_key = llm_api_key or config.LLM_API_KEY
    _base_url = llm_base_url or config.LLM_BASE_URL
    _provider = llm_provider or config.LLM_PROVIDER
    _api_type = llm_api_type or config.LLM_API_TYPE
    _model_id = llm_model_id or config.LLM_MODEL_ID

    registry = ProviderRegistry()
    registry.register(
        _provider,
        AnthropicProvider(api_key=_api_key, base_url=_base_url),
    )

    import uuid
    _session_id = f"{config.SESSION_ID}-{uuid.uuid4().hex[:8]}"

    agent = Agent(
        stream_fn=create_agent_stream_fn(registry),
        session_id=_session_id,
    )

    _max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS

    model = Model(
        id=_model_id,
        provider=_provider,
        api=_api_type,
        base_url=_base_url,
    )
    agent.set_model(model)
    _max_tokens_registry[_session_id] = _max_tokens

    prompt = system_prompt or config.SYSTEM_PROMPT
    suffix = profile.prompt_suffix if not disable_search else ""
    agent.set_system_prompt(prompt + suffix)

    if disable_search:
        tools = list(extra_tools) if extra_tools else []
    else:
        tools = build_all_tools()
        if extra_tools:
            tools.extend(extra_tools)
        tool_names = set(profile.tools) | {t.name for t in (extra_tools or [])}
        tools = [t for t in tools if t.name in tool_names]

    agent.set_tools(tools)

    handler = on_event or globals()["on_event"]
    agent.subscribe(handler)

    return MMAgent(agent)


async def ask(
    question: str,
    *,
    depth: str = "standard",
    **kwargs: Any,
) -> str:
    """One-shot query. Returns the final answer text."""
    agent = create_agent(depth=depth, **kwargs)
    return await agent.prompt(question)
