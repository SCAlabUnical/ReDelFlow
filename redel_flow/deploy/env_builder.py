"""Environment variable builders for ReDel Docker containers.

Constructs the env var dicts injected into each container at runtime.
Each container receives a tailored view of the network — addresses are
computed from that container's perspective using container_address.
"""

import json

from redel_flow.config.configurations import google_api_key, tavily_api_key
from redel_flow.deploy.network import container_address


# ── Network info ──────────────────────────────────────────────────────────────

def build_network_info_for(
    container_host: str,
    agent_lookup: dict,
    orchestrator_cfg: dict,
    client_cfg: dict,
) -> dict:
    """Build the address map from the perspective of a specific container.

    Each container gets addresses correct from its own network perspective —
    same-host containers use host.docker.internal instead of the external IP.
    """
    network_info = {
        name: container_address(
            container_host,
            cfg.get("host", "localhost"),
            cfg["port"]
        )
        for name, cfg in agent_lookup.items()
    }
    network_info["Orchestrator"] = container_address(
        container_host,
        orchestrator_cfg["host"],
        orchestrator_cfg["port"]
    )
    network_info["Client"] = container_address(
        container_host,
        client_cfg["host"],
        client_cfg["port"]
    )
    return network_info


# ── Neighbors ─────────────────────────────────────────────────────────────────

def build_full_neighbors(agent_cfg: dict, agent_lookup: dict) -> dict:
    """Return neighbor config as a JSON-serializable dict.

    Values are lists (not tuples) because they will be JSON-serialized.
    run_single_agent.py converts them back to tuples on the other side.
    Addresses are computed from the agent's own host perspective.
    """
    source_host = agent_cfg.get("host", "localhost")
    neighbors = {}
    for neighbor_name in agent_cfg.get("neighbors", []):
        neighbor_cfg = agent_lookup[neighbor_name]
        address = container_address(
            source_host,
            neighbor_cfg.get("host", "localhost"),
            neighbor_cfg["port"]
        )
        neighbors[neighbor_name] = [
            address,
            neighbor_cfg["role"],
            neighbor_cfg.get("description", ""),
        ]
    return neighbors


# ── Env var builders ──────────────────────────────────────────────────────────

def _base_env(network_info: dict) -> dict:
    """Env vars common to every container."""
    return {
        "NETWORK_INFO":   json.dumps(network_info),
        "GOOGLE_API_KEY": google_api_key,
        "TAVILY_API_KEY": tavily_api_key,
    }


def build_agent_env(
    agent_cfg: dict,
    agent_lookup: dict,
    orchestrator_cfg: dict,
    client_cfg: dict,
) -> dict:
    """Build the full env var dict for an agent container."""
    source_host = agent_cfg.get("host", "localhost")
    neighbors = build_full_neighbors(agent_cfg, agent_lookup)
    network_info = build_network_info_for(
        source_host, agent_lookup, orchestrator_cfg, client_cfg
    )
    return {
        **_base_env(network_info),
        "AGENT_TYPE":        "agent",
        "AGENT_NAME":        agent_cfg["name"],
        "AGENT_ROLE":        agent_cfg["role"],
        "AGENT_DESCRIPTION": agent_cfg.get("description", ""),
        "AGENT_PORT":        str(agent_cfg["port"]),
        "NEIGHBORS":         json.dumps(neighbors),
        "TOOLS":             json.dumps(agent_cfg.get("tools", {})),
    }


def build_orchestrator_env(
    entrypoints: list,
    agent_lookup: dict,
    orchestrator_cfg: dict,
    client_cfg: dict,
) -> dict:
    """Build the full env var dict for the orchestrator container."""
    orch_host = orchestrator_cfg["host"]
    orch_port = int(orchestrator_cfg["port"])

    neighbors = {}
    for entry in entrypoints:
        entry_cfg = agent_lookup[entry]
        address = container_address(
            orch_host,
            entry_cfg.get("host", "localhost"),
            entry_cfg["port"]
        )
        neighbors[entry] = [
            address,
            entry_cfg["role"],
            entry_cfg.get("description", ""),
        ]

    network_info = build_network_info_for(
        orch_host, agent_lookup, orchestrator_cfg, client_cfg
    )

    return {
        **_base_env(network_info),
        "AGENT_TYPE":        "orchestrator",
        "AGENT_NAME":        "Orchestrator",
        "AGENT_ROLE":        "Client Interaction Handler",
        "AGENT_DESCRIPTION": "Receives user requests and delegates to agents.",
        "AGENT_PORT":        str(orch_port),
        "NEIGHBORS":         json.dumps(neighbors),
        "TOOLS":             "{}",
    }