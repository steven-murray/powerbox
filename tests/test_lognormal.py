"""Tests that prove that LogNormalPowerBox produces log-normal fields.

Note that we don't have to show here that the fields have the right power spectrum,
that is done in the roundtrip tests. THis test module is all about showing that the
differences between LogNormalPowerBox and PowerBox are consistent with the log-normal
transformation.
"""

from functools import partial

import numpy as np
import pytest
from scipy.stats import normaltest

from powerbox import LogNormalPowerBox, get_power

get_power = partial(get_power, bins_upto_boxlen=True)


def nice_pk(amp: float):
    """Create a nice P(k) that goes to zero at k=0 and k=infinity."""

    def pkfunc(k):
        return amp / (k**2 + 1 / k)

    return pkfunc


@pytest.mark.parametrize("ncells", [128, 129])
@pytest.mark.parametrize("amp", [0.1, 1.0, 10.0, 100.0])
def test_lognormal_returns_log_normal_densities(ncells, amp):
    pb = LogNormalPowerBox(N=ncells, pk=nice_pk(amp), dim=3, seed=1234, boxlength=100.0)
    densities = np.log(pb.delta_x() + 1)  # log(1 + delta) should be Gaussian
    mean = np.mean(densities)

    assert np.isclose(mean, 0, atol=1e-2), (
        f"Mean of log(1 + delta) should be close to zero, but got {mean}"
    )
    _, p = normaltest(densities.flatten())
    assert p > 0.05, f"Log(1 + delta) should be normally distributed, but normaltest p-value is {p}"
