"""Log streaming for ReDel Docker containers.

Streams container logs to stdout in real time, one daemon thread
per container. Works for both local and remote containers.
"""

import logging
import subprocess
import threading

import docker

logger = logging.getLogger(__name__)


def stream_local_logs(container_name: str):
    """Stream logs from a local container to stdout."""
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        for log in container.logs(stream=True, follow=True):
            line = log.decode(errors="replace").strip()
            if line:
                print(f"[{container_name}] {line}", flush=True)
    except Exception as e:
        logger.error(f"[logs] Stream error for '{container_name}': {e}")


def stream_remote_logs(container_name: str, host: str, ssh_cfg: dict):
    """Stream logs from a remote container via SSH."""
    user = ssh_cfg["ssh_user"]
    key = ssh_cfg["ssh_key"]
    try:
        proc = subprocess.Popen(
            [
                "ssh", "-i", key,
                "-o", "StrictHostKeyChecking=no",
                f"{user}@{host}",
                f"docker logs -f {container_name}"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for line in proc.stdout:
            print(
                f"[{container_name}@{host}] {line.decode(errors='replace').strip()}",
                flush=True
            )
    except Exception as e:
        logger.error(f"[logs] Stream error for '{container_name}@{host}': {e}")


def start_log_streaming(
    agent_configs: list,
    hosts_cfg: dict,
    orchestrator_cfg: dict,
):
    """Start a daemon thread per container to stream all logs to stdout."""
    from redel_flow.deploy.network import is_local

    targets = [
        (cfg["name"], cfg.get("host", "localhost"))
        for cfg in agent_configs
    ] + [("Orchestrator", orchestrator_cfg.get("host", "localhost"))]

    for name, host in targets:
        if is_local(host):
            target = stream_local_logs
            args = (name,)
        else:
            target = stream_remote_logs
            args = (name, host, hosts_cfg[host])

        threading.Thread(target=target, args=args, daemon=True).start()

    logger.info(f"[logs] ✅ Streaming logs for {len(targets)} containers.")