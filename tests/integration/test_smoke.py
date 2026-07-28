from pathlib import Path

import pandas as pd

from chronocline.config import load_config
from chronocline.experiments import run
from chronocline.plotting import plot_capacity
from chronocline.results import validate_result_directory


def test_dry_run_and_result_manifest(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke.yaml"))
    experiment = config.experiment.model_copy(
        update={"output_directory": "results", "require_clean_git": False}
    )
    config = config.model_copy(update={"experiment": experiment})
    plan = run(config, dry_run=True)
    assert plan["jobs"] == 3
    directory = run(config)
    assert not validate_result_directory(directory)
    plot_capacity(pd.read_csv(directory / "results.csv"), directory / "figures")
    # Plot dispatch is tested separately; manifests validate immediately after execution.
