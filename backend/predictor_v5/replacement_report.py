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


def _unlink_owned(directory: int, name: str, identity: tuple[int, int]) -> bool:
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (info.st_dev, info.st_ino) != identity:
        return False
    os.unlink(name, dir_fd=directory)
    return True


def write_report(payload: dict, path: Path) -> dict:
    """Publish canonically without replacement; the parent must already exist."""
    wrapper = _wrapper(payload)
    raw = canonical_bytes(wrapper)
    destination = Path(path)
    directory = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary = f".{destination.name}.tmp.{secrets.token_hex(12)}"
    descriptor = None
    temporary_created = False
    identity = None
    published = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory)
        temporary_created = True
        info = os.fstat(descriptor)
        identity = (info.st_dev, info.st_ino)
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
        if not _unlink_owned(directory, temporary, identity):
            raise OSError("temporary report identity changed during publication")
        temporary_created = False
        os.fsync(directory)
        return wrapper
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if identity is not None and published:
            _unlink_owned(directory, destination.name, identity)
        if identity is not None and temporary_created:
            _unlink_owned(directory, temporary, identity)
        try:
            os.fsync(directory)
        except OSError:
            pass
        raise
    finally:
        os.close(directory)
