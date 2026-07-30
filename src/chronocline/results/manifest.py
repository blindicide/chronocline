"""Schema-2 clean-source provenance and atomic manifest finalization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..constants import SCHEMA_VERSION, __version__

RUNNER_VERSION = "2.0"


def git_state() -> tuple[str | None, bool]:
    """Capture current Git state before output creation."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=all"], text=True
                ).strip()
            ),
        )
    except (OSError, subprocess.CalledProcessError):
        return None, True


def configuration_hash(data: dict[str, Any]) -> str:
    """Hash canonical resolved configuration data."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def run_identifier(config_hash: str, source_commit: str | None, kind: str) -> str:
    """Return code-aware deterministic run identity."""
    payload = [config_hash, source_commit, __version__, SCHEMA_VERSION, kind, RUNNER_VERSION]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def create_manifest(**values: Any) -> dict[str, Any]:
    """Build a manifest prior to writing any output file."""
    values.update(
        {
            "project": "Project Chronocline",
            "package_version": __version__,
            "result_schema_version": SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION,
            "started_at": datetime.now(UTC).isoformat(),
            "completion_status": "running",
            "semantic_validation_status": "pending",
            "semantic_validation_errors": [],
            "generated_files": {},
        }
    )
    return values


def write_environment(directory: Path, manifest: dict[str, Any]) -> None:
    """Write platform and dependency provenance."""
    versions = {}
    for package in ("numpy", "scipy", "pandas", "matplotlib", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "dependencies": versions,
        "package_version": __version__,
        "workers": manifest["workers"],
        "source_commit": manifest["source_commit"],
        "source_dirty": manifest["source_dirty"],
    }
    (directory / "environment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def finalize_manifest(
    directory: Path, manifest: dict[str, Any], completed_jobs: int, semantic_errors: list[str]
) -> None:
    """Checksum generated output then atomically complete the manifest."""
    checksums = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    manifest.update(
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "completion_status": "complete" if not semantic_errors else "failed",
            "completed_jobs": completed_jobs,
            "failed_jobs": (
                0
                if not semantic_errors
                else max(0, int(manifest.get("expected_jobs", 0)) - completed_jobs)
            ),
            "semantic_validation_status": "passed" if not semantic_errors else "failed",
            "semantic_validation_errors": semantic_errors,
            "generated_files": checksums,
        }
    )
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(directory / "manifest.json")


def write_preliminary_manifest(
    directory: Path, manifest: dict[str, Any], completed_jobs: int
) -> None:
    """Persist real work counts before semantic validation consumes the manifest."""
    manifest.update(
        {
            "completed_jobs": completed_jobs,
            "failed_jobs": max(0, int(manifest["expected_jobs"]) - completed_jobs),
            "completion_status": "running",
        }
    )
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
