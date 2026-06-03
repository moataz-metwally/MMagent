# MMagent

Multi-modal research agent built on [PI-agent](https://github.com/iyahi/pi-agent) with Tavily-powered web search tools.

## Features

- **Anthropic Messages API** streaming provider
- **Tavily tools**: search, extract, crawl, map, research, dynamic search
- **Depth profiles**: quick / standard / deep — controls tool selection and thoroughness
- **Per-agent max_tokens**: session-based registry for token limit overrides
- **Pluggable providers**: swap LLM backends via the PI-agent provider interface

## Installation

```bash
pip install pi-agent httpx tavily-python
```

## Configuration

Set environment variables (no hardcoded secrets):

```bash
export LLM_API_KEY="your-api-key"
export LLM_MODEL_ID="claude-sonnet-4-6"
export LLM_BASE_URL="https://api.anthropic.com"
export LLM_PROVIDER="anthropic"
export TAVILY_API_KEY="your-tavily-key"
```

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_ID` | `glm-5.1` | Model identifier |
| `LLM_PROVIDER` | `anthropic` | Provider name |
| `LLM_API_TYPE` | `openai-completions` | API type |
| `LLM_BASE_URL` | `https://api.anthropic.com` | API base URL |
| `LLM_API_KEY` | `""` | LLM API key (required) |
| `LLM_MAX_TOKENS` | `4096` | Default max output tokens |
| `LLM_TEMPERATURE` | `0.7` | Default temperature |
| `TAVILY_API_KEY` | `""` | Tavily API key (required for search tools) |

## Usage

### Quick start

```python
from MMagent import create_agent

agent = create_agent(
    depth="standard",
    system_prompt="You are a research assistant.",
)

response = await agent.prompt("What are the latest developments in quantum computing?")
```

### With extra tools

```python
from MMagent import create_agent
from stock_tools import build_screener_tools

agent = create_agent(
    depth="standard",
    system_prompt="You are a stock analyst.",
    extra_tools=build_screener_tools(),
    disable_search=True,
)

response = await agent.prompt("Analyze AAPL fundamentals")
```

### Depth profiles

```python
# Quick — one search, brief answer
agent = create_agent(depth="quick")

# Standard — search + extract, balanced (default)
agent = create_agent(depth="standard")

# Deep — all tools, exhaustive research
agent = create_agent(depth="deep")
```

## Architecture

```
MMagent
├── agent.py        — Agent factory, session management, max_tokens registry
├── config.py       — Environment-based configuration
├── provider.py     — Anthropic Messages API streaming provider
├── tools.py        — Tavily tool definitions and schemas
├── __init__.py     — Public API exports
└── __main__.py     — CLI entry point
```

## License

MIT
