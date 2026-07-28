"""Traceable result manifest creation and checksum finalization."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..constants import SCHEMA_VERSION, __version__


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_manifest(config_hash: str, experiment: str, seed: int, workers: int, locale: str) -> dict[str, object]:
    """Create provenance metadata before calculation starts."""
    return {"project": "Project Chronocline", "package_version": __version__, "result_schema_version": SCHEMA_VERSION, "experiment": experiment, "started_at": datetime.now(timezone.utc).isoformat(), "completion_status": "running", "git_commit": _git(["git", "rev-parse", "HEAD"]), "git_dirty": bool(_git(["git", "status", "--porcelain"])), "python": platform.python_version(), "platform": platform.platform(), "config_hash": config_hash, "root_seed": seed, "workers": workers, "locale": locale, "generated_files": {}}


def finalize_manifest(directory: Path, manifest: dict[str, object], status: str = "complete") -> None:
    """Atomically write final manifest with checksums of all generated files."""
    checksums = {}
    for path in directory.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            checksums[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.update({"completed_at": datetime.now(timezone.utc).isoformat(), "completion_status": status, "generated_files": checksums})
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(directory / "manifest.json")
