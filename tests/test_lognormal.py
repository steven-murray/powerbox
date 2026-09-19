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
@pytest.mark.parametrize("amp", [0.1, 1.0])
def test_lognormal_returns_log_normal_densities(ncells, amp):
    pb = LogNormalPowerBox(shape=(ncells,) * 3, pk=nice_pk(amp), seed=1234, size=(100.0,) * 3)
    densities = np.log(pb.delta_x() + 1)  # log(1 + delta) should be Gaussian
    _, p = normaltest(densities.flatten())
    assert p > 0.05, f"Log(1 + delta) should be normally distributed, but normaltest p-value is {p}"


def test_lognormal_rejects_power_it_cannot_realize():
    """A field too non-linear for the log-transform raises rather than being approximated.

    The Coles & Jones construction requires log(1 + xi) to be a valid (positive
    semi-definite) correlation function. That fails once the field variance approaches
    unity: here ``amp=10`` gives xi(0) ~ 1.9 and the required Gaussian power spectrum has
    modes as negative as -7% of its largest mode, so no Gaussian field exists whose
    exponential has the requested power spectrum.

    Note that a normality test cannot stand in for this check. Replacing the negative modes
    by their absolute values yields a field that is still perfectly lognormally distributed,
    and only its power spectrum is wrong.
    """
    pb = LogNormalPowerBox(shape=(128,) * 3, pk=nice_pk(10.0), seed=1234, size=(100.0,) * 3)

    assert pb.variance > 1

    with pytest.raises(ValueError, match="not a valid correlation function"):
        pb.delta_x()


@pytest.mark.parametrize(
    ("depth", "expectation"),
    [
        (-4e-5, "warn"),  # a realistic cosmological spectrum on a 256^3 grid
        (-5e-4, "warn"),
        (-2e-2, "raise"),  # a marginal, genuinely too non-linear field
        (-7e-1, "raise"),  # hopeless
    ],
)
def test_positive_definiteness_policy_scales_with_the_size_of_the_violation(depth, expectation):
    """A negligible violation is reported; a material one stops the construction.

    Discretising any realistic power spectrum leaves a few modes of ``log(1 + xi)``
    marginally negative, and zeroing those changes the realized field immeasurably --
    failing there would reject ordinary cosmological use. A field genuinely too non-linear
    for the log-transform is a different thing entirely, and must not be approximated
    silently. The boundary is the depth of the worst negative mode relative to the largest
    mode; the depths parametrized here are measured from real configurations of each kind.
    """
    pb = LogNormalPowerBox(shape=(32,) * 3, pk=nice_pk(0.1), seed=1234, size=(100.0,) * 3)

    gaussian_power = np.asarray(pb.gaussian_power_array())
    assert gaussian_power.min() >= 0
    gaussian_power[(0,) * pb.dim] = depth * gaussian_power.max()

    if expectation == "warn":
        with pytest.warns(UserWarning, match="small violation of the positive-definiteness"):
            pb._validate_gaussian_power(gaussian_power)
    else:
        with pytest.raises(ValueError, match="not a valid correlation function"):
            pb._validate_gaussian_power(gaussian_power)
