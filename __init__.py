"""MMagent — PI-agent + Tavily internet-grounded research agent."""

from .agent import MMAgent, ask, create_agent, extract_final_text, on_event
from .config import DEPTH_PROFILES, SYSTEM_PROMPT, DepthProfile
from .provider import AnthropicProvider
from .tools import TOOL_SCHEMAS

__all__ = [
    "MMAgent",
    "create_agent",
    "ask",
    "extract_final_text",
    "on_event",
    "AnthropicProvider",
    "DepthProfile",
    "DEPTH_PROFILES",
    "SYSTEM_PROMPT",
    "TOOL_SCHEMAS",
]
