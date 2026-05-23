"""Network utilities for the ReDel deploy system.

Provides host resolution and address calculation for Docker containers,
handling the distinction between local and remote hosts and the
host.docker.internal convention for same-host container communication.
"""

import socket

import psutil


def get_all_local_ips() -> set:
    """Detect all IP addresses of this machine across all network interfaces.

    Includes loopback, LAN, VPN and Tailscale interfaces — any IP that
    belongs to this machine is considered local.
    """
    ips = {"localhost", "127.0.0.1"}
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                ips.add(addr.address)
    return ips


LOCAL_IDENTIFIERS: set[str] = get_all_local_ips()


def is_local(host: str) -> bool:
    """Return True if host refers to this machine."""
    return host in LOCAL_IDENTIFIERS


def container_address(source_host: str, target_host: str, target_port: int) -> str:
    """Compute the correct address from a container's network perspective.

    Containers cannot reach other containers on the same host via the
    host's external IP — Docker does not guarantee hairpin NAT.
    When source and target are on the same physical host, use
    host.docker.internal, which Docker resolves to the host machine.

    Args:
        source_host: Host where the calling container runs.
        target_host: Host where the target service runs.
        target_port: Port of the target service.

    Returns:
        Address string in the form host:port.
    """
    same_host = (
        target_host == source_host or
        (is_local(target_host) and is_local(source_host))
    )
    return (
        f"host.docker.internal:{target_port}"
        if same_host
        else f"{target_host}:{target_port}"
    )