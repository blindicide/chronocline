import json
from pathlib import Path

import pandas as pd

from chronocline.config import load_config
from chronocline.experiments import run
from chronocline.plotting import plot_capacity
from chronocline.results import validate_result_directory
from chronocline.results.manifest import finalize_manifest


def test_dry_run_and_result_manifest(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke.yaml"))
    config.experiment.output_directory = str(tmp_path / "results")
    plan = run(config, dry_run=True)
    assert plan["jobs"] == 3
    directory = run(config)
    assert not validate_result_directory(directory)
    plot_capacity(pd.read_csv(directory / "results.csv"), directory / "figures")
    finalize_manifest(directory, json.loads((directory / "manifest.json").read_text()))
    assert not validate_result_directory(directory)
