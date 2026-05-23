"""Agent entrypoint for Docker containers.

Reads the agent configuration from environment variables injected
by run_topology.py and starts a single gRPC agent server.

This script is the CMD of the Docker image and runs inside every
container — both regular agents and the orchestrator.

Expected environment variables:
    AGENT_TYPE          "agent" | "orchestrator"
    AGENT_NAME          Agent unique name
    AGENT_ROLE          Agent functional role
    AGENT_DESCRIPTION   Human-readable description (optional)
    AGENT_PORT          Port to bind the gRPC server on
    NEIGHBORS           JSON: {name: [address, role, description]}
    NETWORK_INFO        JSON: {name: address} for every node in the network
    TOOLS               JSON: {tool_name: {tip: str}} (optional, agents only)
    GOOGLE_API_KEY      Google API key
    TAVILY_API_KEY      Tavily API key
"""

import asyncio
import json
import logging
import os

from redel_flow.core.redel_node import Agent
from redel_flow.core.llm.prompts import AGENT_SYSTEM_PROMPT, ORCHESTRATOR_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Tool building ─────────────────────────────────────────────────────────────

def build_tools(tools_config: dict):
    """Instantiate tools from the TOOLS env var config.

    Imports are done lazily inside each branch so that missing optional
    dependencies only raise errors if that specific tool is actually requested.
    """
    tools = []
    descriptions = []

    for tool_name, tool_config in tools_config.items():
        tip = tool_config.get("tip", "")

        if tool_name == "search":
            from langchain_tavily import TavilySearch
            if not os.environ.get("TAVILY_API_KEY"):
                 raise EnvironmentError(
                    "This agent uses the 'search' tool but TAVILY_API_KEY is not set in .env."
                )
            tools.append(TavilySearch(
                max_results=3,
                topic="general",
                include_answer=True,
                include_raw_content=False,
            ))
            if tip:
                descriptions.append(f"- **search**: {tip}")

        elif tool_name == "python":
            from langchain_experimental.utilities import PythonREPL
            from langchain_core.tools import Tool
            repl = PythonREPL()
            tools.append(Tool(
                name="python_repl",
                description="A Python shell for executing Python code.",
                func=repl.run,
            ))
            if tip:
                descriptions.append(f"- **python_repl**: {tip}")

        elif tool_name == "arxiv_search":
            from langchain_community.tools.arxiv.tool import ArxivQueryRun
            from langchain_community.utilities import ArxivAPIWrapper
            wrapper = ArxivAPIWrapper(load_max_docs=3, top_k_results=1)
            tools.append(ArxivQueryRun(
                api_wrapper=wrapper,
                name="arxiv_search",
                description=(
                    "Searches arXiv for academic papers. "
                    "Supports field prefixes (ti: (title), au: (author), abs: (abstract), cat: (category)) and boolean operators AND, OR, ANDNOT (uppercase). "
                    "To filter by date: 'ti:transformer AND submittedDate:[20240101 TO 20241231]'. "
                    "For simple searches, plain keywords also work: 'graph neural networks'."
                ),
            ))
            if tip:
                descriptions.append(f"- **arxiv_search**: {tip}")

        else:
            logger.warning(f"Unknown tool '{tool_name}' — skipping.")

    tip_section = "\n".join(descriptions)
    return tools, tip_section


# ── Env loading ───────────────────────────────────────────────────────────────

def load_env() -> dict:
    """Read and validate all required environment variables.

    Raises EnvironmentError clearly if anything mandatory is missing,
    so the container fails fast with a readable message.
    """
    required = ["AGENT_NAME", "AGENT_ROLE", "AGENT_PORT", "NETWORK_INFO"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"[run_single_agent] Missing required env vars: {', '.join(missing)}"
        )

    # NEIGHBORS is a JSON dict: {name: [address, role, description]}
    # JSON deserializes tuples as lists — Agent accesses by index so both work
    neighbors_raw = json.loads(os.environ.get("NEIGHBORS", "{}"))
    neighbors = {
        name: tuple(value)
        for name, value in neighbors_raw.items()
    }

    return {
        "type":         os.environ.get("AGENT_TYPE", "agent"),
        "name":         os.environ["AGENT_NAME"],
        "role":         os.environ["AGENT_ROLE"],
        "description":  os.environ.get("AGENT_DESCRIPTION", ""),
        "port":         int(os.environ["AGENT_PORT"]),
        "neighbors":    neighbors,
        "network_info": json.loads(os.environ["NETWORK_INFO"]),
        "tools":        json.loads(os.environ.get("TOOLS", "{}")),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    config = load_env()

    agent_type = config["type"]
    system_prompt = (
        ORCHESTRATOR_SYSTEM_PROMPT
        if agent_type == "orchestrator"
        else AGENT_SYSTEM_PROMPT
    )

    if agent_type == "agent":
        tools, tip_section = build_tools(config["tools"])
    else:
        # Orchestrator has no direct tools
        tools, tip_section = [], ""

    agent = Agent(
        system_prompt=system_prompt,
        name=config["name"],
        role=config["role"],
        description=config["description"],
        host="0.0.0.0",  # always bind on all interfaces inside the container
        port=config["port"],
        neighbors=config["neighbors"],
        tool_tips_section=tip_section,
        tools=tools,
        network_info=config["network_info"],
    )

    logger.info(
        f"[{config['name']}] Starting as {agent_type} on port {config['port']}"
    )
    await agent.serve()


if __name__ == "__main__":
    asyncio.run(main())