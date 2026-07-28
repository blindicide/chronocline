import numpy as np
import pytest

from chronocline.distributions import GaussianMixture, gaussian
from chronocline.quantization import UniformQuantizer


def test_gaussian_moments_and_cdf() -> None:
    law = gaussian(1.0, 2.0)
    assert law.cdf(1.0) == pytest.approx(0.5)
    assert law.mean() == pytest.approx(1.0)
    assert law.variance() == pytest.approx(4.0)


def test_mixture_normalization_is_explicit() -> None:
    with pytest.raises(ValueError):
        GaussianMixture([1, 1], [0, 1], [1, 1])
    assert GaussianMixture([1, 1], [0, 1], [1, 1], normalize=True).weights.sum() == pytest.approx(1)


def test_floor_quantizer_boundaries() -> None:
    q = UniformQuantizer(0.5, 0.25)
    assert q.scalar(0.749) == pytest.approx(0.25)
    assert q.scalar(0.75) == pytest.approx(0.75)
    assert np.array_equal(q.bin_index([0.25, 0.75]), [0, 1])
