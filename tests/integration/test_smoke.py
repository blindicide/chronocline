from pathlib import Path

from chronocline.config import load_config
from chronocline.experiments import run
from chronocline.results import validate_result_directory


def test_dry_run_and_result_manifest(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke.yaml"))
    config.experiment.output_directory = str(tmp_path / "results")
    plan = run(config, dry_run=True)
    assert plan["jobs"] == 3
    directory = run(config)
    assert not validate_result_directory(directory)
