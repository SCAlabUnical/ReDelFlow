"""Docker container lifecycle management for the ReDel deploy system.

Handles building the image, transferring it to remote hosts, starting
and stopping containers both locally and remotely via SSH.

SSH is an implementation detail of remote Docker operations and is
kept private to this module — callers only see Docker-level functions.
"""

import logging
import subprocess
import time

import docker
import docker.errors

logger = logging.getLogger(__name__)

IMAGE_NAME = "redel:latest"


# ── SSH (private) ─────────────────────────────────────────────────────────────

def _ssh_run(host: str, ssh_cfg: dict, command: list, **kwargs):
    """Run a command on a remote host via SSH."""
    user = ssh_cfg["ssh_user"]
    key = ssh_cfg["ssh_key"]
    full_cmd = [
        "ssh", "-i", key,
        "-o", "StrictHostKeyChecking=no",
        f"{user}@{host}",
    ] + command
    return subprocess.run(full_cmd, **kwargs)


# ── Image ─────────────────────────────────────────────────────────────────────

def image_exists_locally() -> bool:
    """Return True if the Docker image already exists on the local daemon."""
    result = subprocess.run(
        ["docker", "images", "-q", IMAGE_NAME],
        capture_output=True, text=True
    )
    return bool(result.stdout.strip())


def build_image():
    """Build the Docker image from the local Dockerfile."""
    logger.info(f"[docker] Building image '{IMAGE_NAME}' ...")
    result = subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, "."],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("[docker] Image build failed.")
    logger.info(f"[docker] ✅ Image '{IMAGE_NAME}' built.")


def _get_remote_image_id(host: str, ssh_cfg: dict) -> str:
    """Return the image ID on a remote host, or empty string if absent."""
    result = _ssh_run(
        host, ssh_cfg,
        ["docker", "inspect", "--format={{.Id}}", IMAGE_NAME],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def transfer_image_if_needed(host: str, ssh_cfg: dict, force: bool = False):
    """Transfer the image to a remote host, skipping if already present."""
    if not force:
        if _get_remote_image_id(host, ssh_cfg):
            logger.info(f"[docker] ✅ Image already present on {host}, skipping transfer.")
            return
    _transfer_image(host, ssh_cfg)


def _transfer_image(host: str, ssh_cfg: dict):
    """Transfer the Docker image via docker save | ssh | docker load."""
    user = ssh_cfg["ssh_user"]
    key = ssh_cfg["ssh_key"]

    logger.info(f"[docker] Transferring image to {host} (this may take a moment)...")

    save_proc = subprocess.Popen(
        ["docker", "save", IMAGE_NAME],
        stdout=subprocess.PIPE,
    )
    load_proc = subprocess.Popen(
        [
            "ssh", "-i", key,
            "-o", "StrictHostKeyChecking=no",
            f"{user}@{host}",
            "docker load",
        ],
        stdin=save_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    save_proc.stdout.close()
    _, stderr = load_proc.communicate()

    if load_proc.returncode != 0:
        raise RuntimeError(
            f"[docker] Image transfer to {host} failed:\n{stderr.decode()}"
        )
    logger.info(f"[docker] ✅ Image transferred to {host}.")


# ── Container lifecycle ───────────────────────────────────────────────────────

def _remove_existing_container(client, name: str):
    """Stop and remove a container if it already exists with the same name."""
    try:
        container = client.containers.get(name)
        logger.warning(f"[docker] Container '{name}' already exists — removing.")
        container.stop()
        container.remove()
    except docker.errors.NotFound:
        pass


def run_local_container(
    name: str, port: int, env_vars: dict
) -> "docker.models.containers.Container":
    """Start a container on the local Docker daemon."""
    client = docker.from_env()
    _remove_existing_container(client, name)
    container = client.containers.run(
        IMAGE_NAME,
        name=name,
        environment=env_vars,
        ports={f"{port}/tcp": port},
        extra_hosts={"host.docker.internal": "host-gateway"},
        detach=True,
    )
    logger.info(f"[docker] ✅ Container '{name}' started locally (id: {container.short_id}).")
    return container


def run_remote_container(
    host: str, ssh_cfg: dict, name: str, port: int, env_vars: dict
):
    """Start a container on a remote host via SSH.

    Env vars are written to a temporary file on the remote machine to
    avoid shell escaping issues and keep values out of process lists.
    The file is removed immediately after the container starts.
    """
    env_file = f"/tmp/redel_{name}.env"
    env_content = "\n".join(f"{k}={v}" for k, v in env_vars.items())

    result = _ssh_run(
        host, ssh_cfg,
        [f"cat > {env_file}"],
        input=env_content.encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"[docker] Could not write env file on {host}: {result.stderr.decode()}"
        )

    _ssh_run(host, ssh_cfg, ["docker", "rm", "-f", name], capture_output=True)

    result = _ssh_run(
        host, ssh_cfg,
        [
            "docker", "run", "-d",
            "--name", name,
            "-p", f"{port}:{port}",
            "--add-host", "host.docker.internal:host-gateway",
            "--env-file", env_file,
            IMAGE_NAME,
        ],
        capture_output=True,
        text=True,
    )

    _ssh_run(host, ssh_cfg, ["rm", "-f", env_file], capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"[docker] Failed to start container '{name}' on {host}:\n{result.stderr}"
        )

    container_id = result.stdout.strip()[:12]
    logger.info(f"[docker] ✅ Container '{name}' started on {host} (id: {container_id}).")


def check_remote_container(host: str, ssh_cfg: dict, name: str, wait: float = 5.0):
    """Verify a remote container is still running after startup."""
    time.sleep(wait)
    result = _ssh_run(
        host, ssh_cfg,
        ["docker", "inspect", "--format={{.State.Status}}", name],
        capture_output=True, text=True
    )
    status = result.stdout.strip()
    if status != "running":
        logs = _ssh_run(
            host, ssh_cfg,
            ["docker", "logs", "--tail=20", name],
            capture_output=True, text=True
        )
        raise RuntimeError(
            f"Container '{name}' on {host} exited immediately (status: {status}).\n"
            f"Last logs:\n{logs.stdout or logs.stderr}"
        )


def stop_remote_container(host: str, ssh_cfg: dict, name: str):
    """Stop and remove a container on a remote host."""
    _ssh_run(host, ssh_cfg, ["docker", "rm", "-f", name], capture_output=True)
    logger.info(f"[docker] 🛑 Container '{name}' stopped on {host}.")


# ── Cleanup ───────────────────────────────────────────────────────────────────

_already_cleaned = False


def cleanup(
    local_containers: list,
    transferred: set,
    hosts_cfg: dict,
    agent_configs: list,
    orchestrator_cfg: dict,
):
    """Stop and remove all started containers, local and remote."""
    global _already_cleaned
    if _already_cleaned:
        return
    _already_cleaned = True

    logger.warning(
        "[docker] ⛔ Stopping all containers... "
        "(this may take a few seconds, do not interrupt)"
    )

    from concurrent.futures import ThreadPoolExecutor

    def stop_local(container):
        try:
            container.stop()
            container.remove()
            logger.info(f"[docker] 🛑 '{container.name}' stopped.")
        except Exception as e:
            logger.error(f"[docker] Error stopping '{container.name}': {e}")

    with ThreadPoolExecutor() as executor:
        for container in local_containers:
            executor.submit(stop_local, container)

        for host in transferred:
            ssh_cfg = hosts_cfg[host]
            for agent_cfg in agent_configs:
                if agent_cfg.get("host") == host:
                    executor.submit(stop_remote_container, host, ssh_cfg, agent_cfg["name"])
            orch_host = orchestrator_cfg.get("host", "localhost")
            if orch_host == host:
                executor.submit(stop_remote_container, host, ssh_cfg, "Orchestrator")