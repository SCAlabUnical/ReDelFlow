"""Topology loading, expansion and validation for the ReDel deploy system.

Handles reading the topology JSON, expanding replica definitions into
concrete agent instances, and validating the configuration before any
deployment attempt.
"""

import json
import logging
import os

from redel_flow.deploy.network import is_local

logger = logging.getLogger(__name__)

LOOPBACK = {"localhost", "127.0.0.1"}


# ── Loading ───────────────────────────────────────────────────────────────────

def load_topology(path: str):
    """Load and expand the topology JSON file.

    Raises FileNotFoundError or json.JSONDecodeError on bad input —
    both are caught and handled with clear messages in __main__.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    for key in ("agents", "entrypoints", "orchestrator", "client"):
        if key not in raw:
            raise ValueError(f"[topology] Missing required key '{key}' in topology file.")

    agent_configs, entrypoints = expand_replicas(raw["agents"], raw["entrypoints"])
    hosts_cfg = raw.get("hosts", {})
    orchestrator_cfg = raw["orchestrator"]
    client_cfg = raw["client"]

    return agent_configs, entrypoints, hosts_cfg, orchestrator_cfg, client_cfg


# ── Expansion ─────────────────────────────────────────────────────────────────

def expand_replicas(agent_configs: list, entrypoints: list):
    """Expand replica definitions into concrete agent instances.

    Handles:
      - Single agents with 'port'
      - Multiple replicas with 'ports' list
      - Per-replica or shared 'host'

    Port conflict detection is intentionally left to validate_topology
    so that errors are reported all at once rather than one at a time.
    """
    base_to_reps: dict[str, list[str]] = {}
    base_to_ports: dict[str, list[int]] = {}
    expanded = []

    for agent in agent_configs:
        base_name = agent["name"]
        replicas = max(1, int(agent.get("replicas", 1)))
        ports_list = agent.get("ports")

        if not ports_list and "port" in agent and replicas == 1:
            ports_list = [int(agent["port"])]

        elif ports_list is not None:
            if len(ports_list) != replicas:
                raise ValueError(
                    f"[topology] '{base_name}': replicas={replicas} "
                    f"but 'ports' has {len(ports_list)} entries."
                )
            try:
                ports_list = [int(p) for p in ports_list]
            except (ValueError, TypeError):
                raise ValueError(
                    f"[topology] '{base_name}': all values in 'ports' must be integers."
                )
        else:
            raise ValueError(
                f"[topology] '{base_name}': missing 'port' (single replica) "
                f"or 'ports' (multiple replicas)."
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
        host_value = agent.get("host", "localhost")

        if isinstance(host_value, list):
            if len(host_value) != len(rep_names):
                raise ValueError(
                    f"[topology] '{base_name}': host list length "
                    f"{len(host_value)} does not match replicas={len(rep_names)}."
                )
            host_list = [str(h) for h in host_value]
        else:
            host_list = [str(host_value)] * len(rep_names)

        for i, rep_name in enumerate(rep_names):
            clone = dict(agent)
            clone["name"] = rep_name
            clone["port"] = port_list[i]
            clone["host"] = host_list[i]
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

def validate_topology(
    agent_configs: list,
    entrypoints: list,
    hosts_cfg: dict,
    orchestrator_cfg: dict,
    client_cfg: dict,
):
    """Validate the expanded topology before any deployment attempt.

    Collects all errors and reports them together so the user can fix
    everything in one pass rather than discovering issues one at a time.

    Raises ValueError with a full list of problems if anything is wrong.
    """
    errors = []
    agent_names = set()
    host_port_pairs = set()

    orch_port = int(orchestrator_cfg.get("port", 0))
    client_port = int(client_cfg.get("port", 0))
    reserved = {orch_port: "Orchestrator", client_port: "Client"}

    # Mixed host check — no loopback mixed with real IPs
    all_hosts = (
        [cfg.get("host", "localhost") for cfg in agent_configs]
        + [orchestrator_cfg.get("host", "localhost")]
        + [client_cfg.get("host", "localhost")]
    )
    has_loopback = any(h in LOOPBACK for h in all_hosts)
    has_real_ip = any(h not in LOOPBACK for h in all_hosts)
    if has_loopback and has_real_ip:
        errors.append(
            "Mixed hosts detected: do not mix 'localhost'/'127.0.0.1' with real IPs. "
            "Use 'localhost' for all (local Docker mode) or real IPs for all (distributed mode)."
        )

    # Client must always be local
    client_host = client_cfg.get("host", "localhost")
    if not is_local(client_host):
        errors.append(
            f"Client host '{client_host}' is not local. "
            f"The client must run on the same machine as run_topology.py. "
            f"Use your local Tailscale IP or 'localhost'."
        )

    # Per-agent checks
    for agent in agent_configs:
        name = agent.get("name", "").strip()
        role = agent.get("role", "").strip()
        host = agent.get("host", "localhost")
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
            except (ValueError, TypeError):
                errors.append(f"Agent '{name}': port must be an integer.")
                port = None

        if port and is_local(host) and port in reserved:
            errors.append(
                f"Agent '{name}': port {port} is reserved for {reserved[port]}."
            )

        if port and host:
            pair = (host, port)
            if pair in host_port_pairs:
                errors.append(
                    f"Agent '{name}': '{host}:{port}' is already used by another agent."
                )
            host_port_pairs.add(pair)

        if not is_local(host) and host not in hosts_cfg:
            errors.append(
                f"Agent '{name}': host '{host}' has no entry in the 'hosts' section."
            )

    # Orchestrator remote check
    orch_host = orchestrator_cfg.get("host", "localhost")
    if not is_local(orch_host) and orch_host not in hosts_cfg:
        errors.append(
            f"Orchestrator: host '{orch_host}' has no entry in the 'hosts' section."
        )

    # Neighbor references
    for agent in agent_configs:
        for neighbor in agent.get("neighbors", []):
            if neighbor not in agent_names:
                errors.append(
                    f"Agent '{agent['name']}': neighbor '{neighbor}' does not exist."
                )

    # Entrypoints
    for entry in entrypoints:
        if entry not in agent_names:
            errors.append(f"Entrypoint '{entry}' does not match any agent name.")

    # SSH config for remote hosts
    for host, cfg in hosts_cfg.items():
        if "ssh_user" not in cfg:
            errors.append(f"Host '{host}': missing 'ssh_user' in hosts config.")
        if "ssh_key" not in cfg:
            errors.append(f"Host '{host}': missing 'ssh_key' in hosts config.")
        key_path = cfg.get("ssh_key", "")
        if key_path and not os.path.isfile(os.path.expanduser(key_path)):
            errors.append(f"Host '{host}': SSH key file not found: '{key_path}'.")

    if errors:
        formatted = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"[topology] Validation failed:\n{formatted}")

    logger.info("[topology] ✅ Topology validation passed.")