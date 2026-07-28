"""Run all checked-in YAML experiment configurations."""

from pathlib import Path

from chronocline.cli import run_suite

if __name__ == "__main__":
    run_suite(Path("configs"))
