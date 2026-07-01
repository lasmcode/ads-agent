# src/ads_agent/infrastructure/mcp/url_validator.py
"""SSRF-safe URL validation for fetch_url and related tools."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket
from urllib.parse import urlparse

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.google",
        "169.254.169.254",
    }
)


class URLValidationError(ValueError):
    """Raised when a URL fails SSRF validation."""


@dataclass(frozen=True, slots=True)
class ValidatedUrl:
    """A URL that passed SSRF checks."""

    url: str
    hostname: str
    scheme: str


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve_and_check_ip(hostname: str) -> None:
    """Resolve hostname and reject if any resolved address is private/reserved."""
    try:
        addr_infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        msg = f"Could not resolve hostname '{hostname}': {exc}"
        raise URLValidationError(msg) from exc

    for info in addr_infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            msg = f"URL hostname '{hostname}' resolves to blocked address {ip_str}"
            raise URLValidationError(msg)


def validate_url(url: str) -> ValidatedUrl:
    """
    Validate a URL for safe outbound fetching.

    Rejects non-http(s) schemes, embedded credentials, localhost,
    private/reserved IPs, and cloud metadata endpoints.
    """
    if not url or not url.strip():
        msg = "URL must not be empty"
        raise URLValidationError(msg)

    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https"):
        msg = f"URL scheme '{parsed.scheme or '(none)'}' is not allowed; use http or https"
        raise URLValidationError(msg)

    if parsed.username or parsed.password:
        msg = "URLs with embedded credentials are not allowed"
        raise URLValidationError(msg)

    hostname = parsed.hostname
    if not hostname:
        msg = "URL must include a hostname"
        raise URLValidationError(msg)

    hostname_lower = hostname.lower().rstrip(".")
    if hostname_lower in _BLOCKED_HOSTNAMES:
        msg = f"URL hostname '{hostname}' is not allowed"
        raise URLValidationError(msg)

    # Direct IP literal check
    try:
        ip = ipaddress.ip_address(hostname_lower.strip("[]"))
        if _is_blocked_ip(ip):
            msg = f"URL points to blocked IP address {hostname}"
            raise URLValidationError(msg)
    except ValueError:
        # Not an IP literal — resolve hostname
        _resolve_and_check_ip(hostname_lower)

    return ValidatedUrl(url=url.strip(), hostname=hostname_lower, scheme=parsed.scheme)
