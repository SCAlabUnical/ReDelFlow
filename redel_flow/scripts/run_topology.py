"""Launch all agents and the orchestrator as Docker containers.

Agents and orchestrator run in isolated containers — local or remote —
configured via environment variables. SSH credentials and host mappings
are read from the topology JSON. API keys come from the local .env file
and are injected into each container at runtime.

Usage:
    python redel_flow/scripts/run_topology.py --topology examples/mytopo.json
    python redel_flow/scripts/run_topology.py --topology examples/mytopo.json --rebuild
"""

import argparse
import json
import logging
import sys
import time

import docker

from redel_flow.deploy.network import LOCAL_IDENTIFIERS, is_local
from redel_flow.deploy.topology import load_topology, validate_topology
from redel_flow.deploy.env_builder import build_agent_env, build_orchestrator_env
from redel_flow.deploy.docker_ops import (
    image_exists_locally,
    build_image,
    transfer_image_if_needed,
    run_local_container,
    run_remote_container,
    check_remote_container,
    cleanup,
)
from redel_flow.deploy.logs import start_log_streaming

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Launch the ReDel agent topology.")
    parser.add_argument("--topology", required=True, help="Path to topology JSON file")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force Docker image rebuild (use when framework code changes)"
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        agent_configs, entrypoints, hosts_cfg, orchestrator_cfg, client_cfg = (
            load_topology(args.topology)
        )
    except FileNotFoundError:
        logger.error(f"❌ Topology file not found: {args.topology}")
        sys.exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"❌ {e}")
        sys.exit(1)

    # ── Validate ──────────────────────────────────────────────────────────────
    try:
        validate_topology(agent_configs, entrypoints, hosts_cfg, orchestrator_cfg, client_cfg)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Docker daemon check ───────────────────────────────────────────────────
    try:
        docker.from_env().ping()
    except Exception:
        logger.error("❌ Docker daemon is not running or not accessible.")
        sys.exit(1)

    agent_lookup = {a["name"]: a for a in agent_configs}
    logger.info(f"Local identifiers: {LOCAL_IDENTIFIERS}")

    # ── Build image ───────────────────────────────────────────────────────────
    if args.rebuild or not image_exists_locally():
        try:
            build_image()
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        logger.info("[docker] ✅ Using existing local image (pass --rebuild to force rebuild).")

    # ── Transfer image to remote hosts ────────────────────────────────────────
    transferred: set[str] = set()
    try:
        remote_hosts = {
            cfg["host"] for cfg in agent_configs
            if not is_local(cfg.get("host", "localhost"))
        }
        for host in remote_hosts:
            transfer_image_if_needed(host, hosts_cfg[host], force=args.rebuild)
            transferred.add(host)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Start containers ──────────────────────────────────────────────────────
    local_containers: list = []

    try:
        for agent_cfg in agent_configs:
            host = agent_cfg.get("host", "localhost")
            name = agent_cfg["name"]
            port = agent_cfg["port"]
            env_vars = build_agent_env(agent_cfg, agent_lookup, orchestrator_cfg, client_cfg)

            if is_local(host):
                container = run_local_container(name, port, env_vars)
                local_containers.append(container)
            else:
                run_remote_container(host, hosts_cfg[host], name, port, env_vars)
                check_remote_container(host, hosts_cfg[host], name)

        # Orchestrator
        orch_host = orchestrator_cfg["host"]
        orch_port = int(orchestrator_cfg["port"])
        orch_env = build_orchestrator_env(entrypoints, agent_lookup, orchestrator_cfg, client_cfg)

        if is_local(orch_host):
            orch_container = run_local_container("Orchestrator", orch_port, orch_env)
            local_containers.append(orch_container)
        else:
            if orch_host not in hosts_cfg:
                raise RuntimeError(
                    f"Orchestrator host '{orch_host}' has no entry in 'hosts'."
                )
            if orch_host not in transferred:
                transfer_image_if_needed(orch_host, hosts_cfg[orch_host], force=args.rebuild)
                transferred.add(orch_host)
            run_remote_container(orch_host, hosts_cfg[orch_host], "Orchestrator", orch_port, orch_env)
            check_remote_container(orch_host, hosts_cfg[orch_host], "Orchestrator")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        cleanup(local_containers, transferred, hosts_cfg, agent_configs, orchestrator_cfg)
        sys.exit(1)

    # ── Start log streaming ───────────────────────────────────────────────────
    start_log_streaming(agent_configs, hosts_cfg, orchestrator_cfg)

    logger.info("✅ All containers started.")
    logger.info("Start the client with: python redel_flow/scripts/client.py --topology <path>")
    logger.info("Press Ctrl+C to stop all containers.")

    # ── Keep alive ────────────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(local_containers, transferred, hosts_cfg, agent_configs, orchestrator_cfg)


if __name__ == "__main__":
    main()