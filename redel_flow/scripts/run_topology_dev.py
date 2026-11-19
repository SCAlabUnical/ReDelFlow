"""Development launcher: runs all agents and orchestrator as local processes.

No Docker, no SSH, no infrastructure required.
Host fields in the topology are ignored — everything runs on localhost.
Only ports matter in this mode.

Usage:
    python redel_flow/scripts/run_topology_dev.py --topology examples/mytopo.json
"""

import asyncio
import json
import logging
import os
from multiprocessing import Process

from redel_flow.config.configurations import tavily_api_key, google_api_key
from redel_flow.core.redel_node import Agent
from redel_flow.core.llm.prompts import AGENT_SYSTEM_PROMPT, ORCHESTRATOR_SYSTEM_PROMPT
from langchain_tavily import TavilySearch
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key


# ── Topology loading ──────────────────────────────────────────────────────────

def load_topology(path: str):
    """Load and expand topology. Returns agents, entrypoints, orchestrator and client config."""
    with open(path, "r") as f:
        raw = json.load(f)

    for key in ("agents", "entrypoints", "orchestrator", "client"):
        if key not in raw:
            raise ValueError(f"[topology] Missing required key '{key}'.")

    agent_configs, entrypoints = expand_replicas(raw["agents"], raw["entrypoints"])
    orchestrator_cfg = raw["orchestrator"]
    client_cfg = raw["client"]

    return agent_configs, entrypoints, orchestrator_cfg, client_cfg


def expand_replicas(agent_configs: list, entrypoints: list):
    """Expand replica definitions into concrete agent instances.

    Host is intentionally ignored in dev mode — all agents run on localhost.
    """
    seen_names = set()
    for agent in agent_configs:
        if agent["name"] in seen_names:
            raise ValueError(f"[topology] Duplicate agent name '{agent['name']}'.")
        seen_names.add(agent["name"])

    used_ports = set(int(a["port"]) for a in agent_configs if "port" in a)
    base_to_reps = {}
    base_to_ports = {}
    expanded = []

    for agent in agent_configs:
        base_name = agent["name"]
        replicas = max(1, int(agent.get("replicas", 1)))
        ports_list = agent.get("ports")

        if not ports_list and "port" in agent and replicas == 1:
            ports_list = [int(agent["port"])]
            used_ports.add(ports_list[0])
        elif ports_list is not None:
            if len(ports_list) != replicas:
                raise ValueError(
                    f"[topology] '{base_name}': replicas={replicas} "
                    f"but ports has {len(ports_list)} entries."
                )
            try:
                ports_list = [int(p) for p in ports_list]
            except (ValueError, TypeError):
                raise ValueError(
                    f"[topology] '{base_name}': all values in 'ports' must be integers."
                )
            for port in ports_list:
                if port in used_ports:
                    raise ValueError(f"[topology] Port {port} is already in use.")
                used_ports.add(port)
        else:
            raise ValueError(
                f"[topology] '{base_name}': missing 'port' or 'ports'."
            )

        base_to_ports[base_name] = ports_list
        base_to_reps[base_name] = (
            [base_name] if replicas == 1
            else [f"{base_name}Replica{i}" for i in range(1, replicas + 1)]
        )

    for agent in agent_configs:
        base_name = agent["name"]
        port_list = base_to_ports[base_name]
        rep_names = base_to_reps[base_name]

        for i, rep_name in enumerate(rep_names):
            clone = dict(agent)
            clone["name"] = rep_name
            clone["port"] = port_list[i]
            clone["host"] = "localhost"   # sempre localhost in dev mode
            expanded.append(clone)

    for node in expanded:
        node["neighbors"] = [
            rep
            for n in node.get("neighbors", [])
            for rep in base_to_reps.get(n, [n])
        ]

    expanded_entrypoints = [
        rep
        for e in entrypoints
        for rep in base_to_reps.get(e, [e])
    ]

    return expanded, expanded_entrypoints


# ── Validation ────────────────────────────────────────────────────────────────

def validate_topology(agent_configs: list, entrypoints: list, orchestrator_cfg: dict, client_cfg: dict):
    """Validate topology for dev mode.

    Host fields are not validated — they are ignored.
    Checks names, ports, neighbors and entrypoints.
    """
    errors = []
    agent_names = set()
    used_ports = set()

    orch_port = int(orchestrator_cfg.get("port", 0))
    client_port = int(client_cfg.get("port", 0))
    reserved = {orch_port: "Orchestrator", client_port: "Client"}

    for agent in agent_configs:
        name = agent.get("name", "").strip()
        role = agent.get("role", "").strip()
        port = agent.get("port")

        if not name:
            errors.append("An agent is missing the 'name' field.")
            continue
        if not role:
            errors.append(f"Agent '{name}': missing 'role' field.")

        if name in agent_names:
            errors.append(f"Duplicate agent name: '{name}'.")
        agent_names.add(name)

        if port is None:
            errors.append(f"Agent '{name}': missing 'port' field.")
        else:
            try:
                port = int(port)
                if not (1 <= port <= 65535):
                    errors.append(f"Agent '{name}': port {port} out of range (1–65535).")
                elif port in reserved:
                    errors.append(
                        f"Agent '{name}': port {port} is reserved for {reserved[port]}."
                    )
                elif port in used_ports:
                    errors.append(f"Agent '{name}': port {port} is already in use.")
                else:
                    used_ports.add(port)
            except (ValueError, TypeError):
                errors.append(f"Agent '{name}': port must be an integer.")

    for agent in agent_configs:
        for neighbor in agent.get("neighbors", []):
            if neighbor not in agent_names:
                errors.append(
                    f"Agent '{agent['name']}': neighbor '{neighbor}' does not exist."
                )

    for entry in entrypoints:
        if entry not in agent_names:
            errors.append(f"Entrypoint '{entry}' does not match any agent name.")

    if errors:
        formatted = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"[topology] Validation failed:\n{formatted}")

    logger.info("[run_topology_dev] ✅ Topology validation passed.")


# ── Network info ──────────────────────────────────────────────────────────────

def build_network_info(agent_lookup: dict, orchestrator_cfg: dict, client_cfg: dict) -> dict:
    """Build address map. Everything is localhost in dev mode."""
    network_info = {
        name: f"localhost:{cfg['port']}"
        for name, cfg in agent_lookup.items()
    }
    network_info["Orchestrator"] = f"localhost:{orchestrator_cfg['port']}"
    network_info["Client"] = f"localhost:{client_cfg['port']}"
    return network_info


# ── Agent building ────────────────────────────────────────────────────────────

def build_full_neighbors(agent_cfg: dict, agent_lookup: dict) -> dict:
    neighbors = {}
    for neighbor_name in agent_cfg.get("neighbors", []):
        neighbor_cfg = agent_lookup[neighbor_name]
        neighbors[neighbor_name] = (
            f"localhost:{neighbor_cfg['port']}",
            neighbor_cfg["role"],
            neighbor_cfg.get("description", ""),
        )
    return neighbors


def build_tools(agent_config: dict):
    tools = []
    descriptions = []
    for tool_name, tool_config in agent_config.get("tools", {}).items():
        tip = tool_config.get("tip", "")
        if tool_name == "search":
            if not os.environ.get("TAVILY_API_KEY"):
                raise EnvironmentError(
                    f"Agent '{agent_config['name']}' uses the 'search' tool "
                    f"but TAVILY_API_KEY is not set in .env."
                )
            tools.append(TavilySearch(
                max_results=3, topic="general",
                include_answer=True, include_raw_content=False,
            ))
            if tip:
                descriptions.append(f"- **search**: {tip}")
        elif tool_name == "python":
            repl = PythonREPL()
            tools.append(Tool(
                name="python_repl",
                description="A Python shell for executing Python code.",
                func=repl.run,
            ))
            if tip:
                descriptions.append(f"- **python_repl**: {tip}")
        elif tool_name == "arxiv_search":
            wrapper = ArxivAPIWrapper(load_max_docs=3, top_k_results=1)
            tools.append(ArxivQueryRun(
                api_wrapper=wrapper, name="arxiv_search",
                description=(
                    "Searches arXiv for academic papers. "
                    "Supports field prefixes (ti:, au:, abs:, cat:) and boolean operators AND, OR, ANDNOT (uppercase). "
                    "To filter by date: 'ti:transformer AND submittedDate:[202401010000 TO 202412312359]'. "
                    "For simple searches, plain keywords also work: 'graph neural networks'."
                ),
            ))
            if tip:
                descriptions.append(f"- **arxiv_search**: {tip}")
    tip_section = "\n".join(descriptions)
    return tools, tip_section


# ── Process entry points ──────────────────────────────────────────────────────

def start_agent(agent_config: dict, agent_lookup: dict, network_info: dict):
    neighbors = build_full_neighbors(agent_config, agent_lookup)
    tools, tip_section = build_tools(agent_config)
    agent = Agent(
        AGENT_SYSTEM_PROMPT,
        agent_config["name"],
        agent_config["role"],
        agent_config.get("description", ""),
        "localhost",
        agent_config["port"],
        neighbors,
        tip_section,
        tools=tools,
        network_info=network_info,
    )
    try:
        asyncio.run(agent.serve())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"[{agent_config['name']}] ❌ UNRECOVERABLE ERROR: {e}")


def start_orchestrator(entrypoints: list, agent_lookup: dict, network_info: dict, orchestrator_cfg: dict):
    port = int(orchestrator_cfg["port"])
    neighbors = {}
    for entry in entrypoints:
        entry_cfg = agent_lookup[entry]
        neighbors[entry] = (
            f"localhost:{entry_cfg['port']}",
            entry_cfg["role"],
            entry_cfg.get("description", ""),
        )
    agent = Agent(
        ORCHESTRATOR_SYSTEM_PROMPT,
        "Orchestrator",
        "Client Interaction Handler",
        "Receives user requests and delegates to agents.",
        "localhost",
        port,
        neighbors,
        "",
        tools=[],
        network_info=network_info,
    )
    try:
        asyncio.run(agent.serve())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"[Orchestrator] ❌ UNRECOVERABLE ERROR: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch ReDel topology as local processes (dev mode, no Docker)."
    )
    parser.add_argument("--topology", required=True, help="Path to topology JSON file")
    args = parser.parse_args()

    try:
        agent_configs, entrypoints, orchestrator_cfg, client_cfg = load_topology(args.topology)
    except FileNotFoundError:
        logger.error(f"[run_topology_dev] ❌ Topology file not found: {args.topology}")
        raise SystemExit(1)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"[run_topology_dev] ❌ {e}")
        raise SystemExit(1)

    try:
        validate_topology(agent_configs, entrypoints, orchestrator_cfg, client_cfg)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    agent_lookup = {a["name"]: a for a in agent_configs}
    network_info = build_network_info(agent_lookup, orchestrator_cfg, client_cfg)

    logger.info(f"[run_topology_dev] Network info: {network_info}")

    processes = []

    for agent_cfg in agent_configs:
        p = Process(target=start_agent, args=(agent_cfg, agent_lookup, network_info))
        p.start()
        processes.append(p)
        logger.info(f"[run_topology_dev] ✅ {agent_cfg['name']} started on port {agent_cfg['port']}")

    p = Process(target=start_orchestrator, args=(entrypoints, agent_lookup, network_info, orchestrator_cfg))
    p.start()
    processes.append(p)
    logger.info(f"[run_topology_dev] ✅ Orchestrator started on port {orchestrator_cfg['port']}")

    logger.info("[run_topology_dev] ✅ All agents started.")
    logger.info("[run_topology_dev] Start the client with: python redel_flow/scripts/client.py")
    logger.info("[run_topology_dev] Press Ctrl+C to terminate.")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.warning("[run_topology_dev] ⛔ Stopping all processes.")
        for p in processes:
            p.terminate()