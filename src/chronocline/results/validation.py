"""Result directory integrity validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def validate_result_directory(path: str | Path) -> list[str]:
    """Return integrity errors; an empty list means the manifest validates."""
    directory = Path(path)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists() or not (directory / "results.csv").exists():
        return ["missing manifest.json or results.csv"]
    manifest = json.loads(manifest_path.read_text())
    errors = []
    for name, expected in manifest.get("generated_files", {}).items():
        file = directory / name
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            errors.append(f"checksum mismatch: {name}")
    return errors
