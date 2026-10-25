"""Canonical, durable, no-replace replacement report publication."""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path

from predictor_v5.replacement_contract import canonical_bytes


def _wrapper(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("report payload must be an object")  # noqa: TRY004
    payload_bytes = canonical_bytes(payload)
    return {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }


def _render_report(payload: dict) -> bytes:
    """Return the exact immutable bytes accepted by the lifecycle renderer."""
    return canonical_bytes(_wrapper(payload))


def _write_all(descriptor: int, raw: bytes) -> None:
    remaining = memoryview(raw)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("report write made no progress")
        remaining = remaining[written:]


def write_report(payload: dict, path: Path) -> dict:
    """Atomically publish one canonical report without replacing a report."""
    wrapper = _wrapper(payload)
    raw = canonical_bytes(wrapper)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary = f".{destination.name}.tmp.{secrets.token_hex(12)}"
    descriptor = None
    published = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory)
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary,
                destination.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(f"replacement report already exists: {destination}") from exc
        published = True
        os.unlink(temporary, dir_fd=directory)
        temporary = ""
        os.fsync(directory)
        return wrapper
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if published:
            try:
                os.unlink(destination.name, dir_fd=directory)
            except FileNotFoundError:
                pass
        if temporary:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
        try:
            os.fsync(directory)
        except OSError:
            pass
        raise
    finally:
        os.close(directory)
