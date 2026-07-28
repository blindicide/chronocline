import pytest
from pydantic import ValidationError

from chronocline.config import RunConfig


def test_configuration_rejects_bad_step() -> None:
    data = {
        "experiment": {"name": "x"},
        "channel": {"alphabet": {"values": [0, 1]}},
        "jitter": {"distribution": "gaussian"},
        "quantizer": {"step": -1},
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)
