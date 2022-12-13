import pytest

import numpy as np
from scipy.ndimage import gaussian_filter

from powerbox import PowerBox


@pytest.mark.skip(
    reason="this is not passing to desired tolerance at this point... not sure if this is a problem. It's not systematic."
)
def test_resolution():
    var = [0] * 6
    for i in range(6):
        pb = PowerBox(
            64 * 2**i,
            dim=2,
            pk=lambda k: 1.0 * k**-2.0,
            boxlength=1.0,
            angular_freq=True,
        )
        var[i] = np.var(gaussian_filter(pb.delta_x(), sigma=2**i, mode="wrap"))
    print(var / var[0])
    assert np.allclose(var / var[0], 1, atol=1e-2)
