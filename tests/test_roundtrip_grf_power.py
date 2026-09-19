"""Physical tests of the full end-to-end process of field generation and power recovery.

This module tests the combination of PowerBox (and its subclasses) and get_power.

Fourier-convention handling is *not* tested here: it is covered exactly, to machine
precision, in ``test_convention_equivalence.py``. This module runs at the default
convention and instead spends its budget on statistical sensitivity, so that it can see a
small systematic bias in the recovered power rather than only a gross one.
"""

import numpy as np
import pytest
from scipy import stats

from powerbox import LogNormalPowerBox, PowerBox, dft, get_power
from powerbox.tools import _magnitude_grid, angular_average

# Chosen so the test detects a uniform 2% bias in the recovered power with a wide margin
# (p ~ 3e-10), while leaving an unbiased roundtrip consistent (chi2/dof ~ 1.2, p ~ 0.4).
# Two thirds of this detects 2% only marginally (p ~ 1e-4). A percent-level sensitivity is
# what makes this test worth running: a normalisation error affecting one surface of the
# Hermitian spectrum shows up as only a ~1.5% median deficit here (though ~11% in the
# lowest-|k| modes), so a coarser test would pass straight through it.
NREALIZATIONS = 192

# A uniform bias of this size must be detected; asserted directly by
# test_roundtrip_statistic_detects_a_small_bias.
DETECTABLE_BIAS = 0.02


def expected_binned_power(pb: PowerBox, psobj, pk) -> np.ndarray:
    """Return the exact expectation of the binned power: the mean of ``pk`` over each bin.

    Comparing against ``pk(bin_avg)`` instead would introduce a deterministic bias
    wherever ``pk`` is curved across a bin, which is largest in the sparsely-populated
    low-|k| bins. Averaging the theory through the same binning as the measurement removes
    that by construction, so the null hypothesis under test is exactly right.
    """
    freq = [dft.fftfreq(n, d=d, b=pb.fourier_b) for n, d in zip(pb.shape, pb.dx, strict=True)]
    kmag = _magnitude_grid(list(freq))
    nonzero = kmag > 0
    theory = np.where(nonzero, pk(np.where(nonzero, kmag, 1)), np.nan)
    return angular_average(theory, freq, psobj.bin_edges, bins_upto_boxlen=True)[0]


def roundtrip_pvalue(
    measured_power: np.ndarray,
    expected_power: np.ndarray,
    nrealizations: int,
) -> tuple[float, int, float]:
    """Return the two-sided p-value of a chi-square test of ensemble power recovery.

    Each bin contributes a t-score: the deviation of the ensemble-mean power from theory,
    in units of the empirical standard error of that mean. Adjacent radial bins are
    correlated, so the series is thinned to approximately independent samples before the
    scores are combined.
    """
    mean_power = np.mean(measured_power, axis=0)
    std_power = np.std(measured_power, axis=0, ddof=1)

    mask = (
        np.isfinite(mean_power)
        & np.isfinite(std_power)
        & (std_power > 0)
        & np.isfinite(expected_power)
        & (expected_power > 0)
    )
    # Exclude the zero mode explicitly.
    mask[0] = False
    assert np.count_nonzero(mask) >= 8

    tscore = (mean_power[mask] - expected_power[mask]) / (std_power[mask] / np.sqrt(nrealizations))
    stride = max(1, tscore.size // 32)
    tscore = tscore[::stride]

    dof = tscore.size
    chi2 = float(np.sum(tscore**2))
    cdf = stats.chi2.cdf(chi2, dof)
    return 2 * min(cdf, 1 - cdf), dof, chi2 / dof


def pk_like_matter_power(amp=1, low_slope=1, hi_slope=2):
    def pk(k):
        return amp / (k**hi_slope + 1 / k**low_slope)

    return pk


def realize_power(boxtype, shape, size, pkfunc, nrealizations=NREALIZATIONS):
    """Return the stacked recovered power and its exact expectation."""
    power = []
    for seed in range(nrealizations):
        pb = boxtype(shape=shape, pk=pkfunc, size=size, ensure_physical=False, seed=seed)
        psobj = get_power(pb.delta_x(), pb.size, bins_upto_boxlen=True)
        power.append(psobj.power)

    return np.asarray(power), expected_binned_power(pb, psobj, pkfunc)


@pytest.mark.parametrize(
    ("shape", "size"),
    [
        ((48, 72), (120.0, 180.0)),
        ((49, 72), (120.0, 180.0)),
        ((48, 71), (120.0, 180.0)),
        ((49, 71), (120.0, 180.0)),
        ((32, 40, 48), (80.0, 100.0, 120.0)),
    ],
)
@pytest.mark.parametrize("boxtype", [PowerBox, LogNormalPowerBox])
def test_roundtrip_power_recovery(shape, size, boxtype) -> None:
    """Non-cubic fields recover their input power spectrum, across odd/even grids."""
    pkfunc = pk_like_matter_power(amp=1, low_slope=1, hi_slope=2)
    power, expected = realize_power(boxtype, shape, size, pkfunc)

    pvalue, dof, reduced_chi2 = roundtrip_pvalue(power, expected, NREALIZATIONS)
    assert pvalue > 1e-4, (
        f"roundtrip inconsistency: dof={dof}, chi2/dof={reduced_chi2:.3f}, p={pvalue:.3e}"
    )


@pytest.mark.parametrize("boxtype", [PowerBox, LogNormalPowerBox])
def test_roundtrip_statistic_detects_a_small_bias(boxtype) -> None:
    """The roundtrip test is powerful enough to be a meaningful check.

    A test that passes is only informative if it would fail on a realistic defect. This
    asserts the sensitivity claimed by ``NREALIZATIONS``: a uniform bias of
    ``DETECTABLE_BIAS`` in the recovered power is rejected, while the true power is not.
    """
    shape, size = (48, 72), (120.0, 180.0)
    pkfunc = pk_like_matter_power(amp=1, low_slope=1, hi_slope=2)
    power, expected = realize_power(boxtype, shape, size, pkfunc)

    unbiased, _, _ = roundtrip_pvalue(power, expected, NREALIZATIONS)
    assert unbiased > 1e-4

    biased, _, _ = roundtrip_pvalue(power * (1 + DETECTABLE_BIAS), expected, NREALIZATIONS)
    assert biased < 1e-4, (
        f"a {DETECTABLE_BIAS:.0%} bias was not detected (p={biased:.3e}); the roundtrip "
        "test is too weak to guard against normalisation errors."
    )
