"""Physical tests of the full end-to-end process of field generation and power recovery.

This module tests the combination of PowerBox (and its subclasses) and get_power.
"""

import numpy as np
import pytest
from scipy import stats

from powerbox import LogNormalPowerBox, PowerBox, get_power


def _assert_reasonable_power_recovery(
    measured_power: np.ndarray,
    expected_power: np.ndarray,
    nsamples: np.ndarray,
    nrealizations: int,
) -> None:
    """Require statistically consistent roundtrip recovery via a t/chi-square test.

    The test compares the ensemble mean of recovered power to theory, normalized by the
    empirical standard error in each bin. Adjacent radial bins are correlated, so we
    thin the t-series before applying a chi-square goodness-of-fit test.
    """
    mean_power = np.mean(measured_power, axis=0)
    std_power = np.std(measured_power, axis=0, ddof=1)

    mask = (
        np.isfinite(mean_power)
        & np.isfinite(std_power)
        & (std_power > 0)
        & np.isfinite(expected_power)
        & (expected_power > 0)
        & (nsamples >= 4)
    )

    # Exclude the zero mode explicitly.
    mask[0] = False
    assert np.count_nonzero(mask) >= 8

    tscore = (mean_power[mask] - expected_power[mask]) / (std_power[mask] / np.sqrt(nrealizations))

    # Neighboring bins are correlated; thin to approximately independent samples.
    stride = max(1, tscore.size // 32)
    tscore = tscore[::stride]

    dof = tscore.size
    chi2 = np.sum(tscore**2)
    cdf = stats.chi2.cdf(chi2, dof)
    two_sided_p = 2 * min(cdf, 1 - cdf)
    assert two_sided_p > 1e-4, (
        f"roundtrip inconsistency: dof={dof}, chi2/dof={chi2 / dof:.3f}, p={two_sided_p:.3e}"
    )


def pk_like_matter_power(amp=1, low_slope=1, hi_slope=2):
    def pk(k):
        return amp / (k**hi_slope + 1 / k**low_slope)

    return pk


@pytest.mark.parametrize(
    ("shape", "size", "a", "b", "pkamp", "low_slope", "hi_slope"),
    [
        ((48, 72), (120.0, 180.0), 1, 1, 1, 1, 2),
        ((49, 72), (120.0, 180.0), 1, 1, 1, 1, 2),
        ((48, 71), (120.0, 180.0), 1, 1, 1, 1, 2),
        ((49, 71), (120.0, 180.0), 1, 1, 1, 1, 2),
        ((49, 71), (120.0, 180.0), 0, 1, 1, 1, 2),
        ((49, 71), (120.0, 180.0), 0, 2 * np.pi, 1, 1, 2),
        ((49, 71), (120.0, 180.0), 1, 2 * np.pi, 1, 1, 2),
    ],
)
@pytest.mark.parametrize("boxtype", [PowerBox, LogNormalPowerBox])
def test_roundtrip_power_recovery(shape, size, boxtype, a, b, pkamp, low_slope, hi_slope) -> None:
    """Non-cubic fields recover input power across odd/even grids and Fourier conventions."""
    nrealizations = 24
    power = []

    pkfunc = pk_like_matter_power(amp=pkamp, low_slope=low_slope, hi_slope=hi_slope)
    for seed in range(nrealizations):
        pb = boxtype(
            shape=shape,
            pk=pkfunc,
            size=size,
            ensure_physical=False,
            seed=seed,
            a=a,
            b=b,
        )
        psobj = get_power(pb.delta_x(), pb.size, a=a, b=b, bins_upto_boxlen=True)
        power.append(psobj.power)

    power = np.asarray(power)
    expected = pkfunc(psobj.bin_avg)
    _assert_reasonable_power_recovery(
        measured_power=power,
        expected_power=expected,
        nsamples=psobj.nsamples,
        nrealizations=nrealizations,
    )
