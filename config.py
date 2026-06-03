"""MMagent — all configuration in one place."""

import os
from dataclasses import dataclass

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "glm-5.1")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_API_TYPE = os.getenv("LLM_API_TYPE", "openai-completions")
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://api.anthropic.com",
)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# ── Tavily ────────────────────────────────────────────────────────────────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Depth profiles ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DepthProfile:
    tools: list[str]
    search_depth: str
    max_results: int
    include_raw_content: bool
    extract_depth: str
    crawl_max_depth: int
    crawl_max_breadth: int
    crawl_limit: int
    map_max_depth: int
    map_limit: int
    research_model: str
    prompt_suffix: str


DEPTH_PROFILES: dict[str, DepthProfile] = {
    "quick": DepthProfile(
        tools=["tavily_search"],
        search_depth="fast",
        max_results=3,
        include_raw_content=False,
        extract_depth="basic",
        crawl_max_depth=1,
        crawl_max_breadth=10,
        crawl_limit=10,
        map_max_depth=1,
        map_limit=20,
        research_model="mini",
        prompt_suffix=(
            "\n\nDEPTH MODE: quick. Answer fast. Use only one search call. "
            "Do not chain tools. Give a brief, direct answer."
        ),
    ),
    "standard": DepthProfile(
        tools=["tavily_search", "tavily_extract", "tavily_dynamic_search"],
        search_depth="basic",
        max_results=10,
        include_raw_content=False,
        extract_depth="advanced",
        crawl_max_depth=2,
        crawl_max_breadth=100,
        crawl_limit=50,
        map_max_depth=2,
        map_limit=50,
        research_model="auto",
        prompt_suffix=(
            "\n\nDEPTH MODE: standard. Use search and extract as needed. "
            "Chain up to 2 tool calls if necessary. Give a balanced answer."
        ),
    ),
    "deep": DepthProfile(
        tools=[
            "tavily_search",
            "tavily_extract",
            "tavily_crawl",
            "tavily_map",
            "tavily_research",
            "tavily_dynamic_search",
        ],
        search_depth="advanced",
        max_results=10,
        include_raw_content=True,
        extract_depth="advanced",
        crawl_max_depth=3,
        crawl_max_breadth=30,
        crawl_limit=100,
        map_max_depth=3,
        map_limit=100,
        research_model="pro",
        prompt_suffix=(
            "\n\nDEPTH MODE: deep. Be exhaustive. Chain multiple tool calls. "
            "Search broadly, extract key URLs, crawl relevant sites, run research. "
            "Provide a thorough, well-cited answer."
        ),
    ),
}

# ── Agent behaviour ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a deep-research assistant with powerful web search, extraction, "
    "crawling, mapping, and research tools from Tavily. "
    "IMPORTANT — search is expensive. Before calling any search tool, check if "
    "the information you need is already available in the conversation context. "
    "Only search when you genuinely need fresh, updated, or missing information "
    "that you cannot infer from existing data. "
    "When you do search, decide which tools to call (you may call several in "
    "sequence) to gather the best information. "
    "Always cite your sources. If one tool is not enough, chain them — e.g. "
    "search first, then extract key URLs, or map a site then crawl specific pages. "
    "Be thorough but concise in your final answer."
)

SESSION_ID = "mmagent"
