"""Physical tests of the full end-to-end process of field generation and power recovery.

This module tests the combination of PowerBox (and its subclasses) and get_power.
"""

import numpy as np
import pytest

from powerbox import LogNormalPowerBox, PowerBox, get_power


def _assert_reasonable_power_recovery(zscore: np.ndarray) -> None:
    """Require that power-recovery z-scores stay near 3-sigma overall."""
    frac_above_three_sigma = np.count_nonzero(zscore > 3.0) / zscore.size
    assert frac_above_three_sigma <= 0.1, zscore
    assert np.max(zscore) < 5.0, zscore


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
        ((8001,), (15.0,), 1, 1, 1, 1, 2),
        ((8000,), (15.0,), 1, 1, 1, 1, 2),
        ((8000,), (15.0,), 1, 2 * np.pi, 1, 1, 2),
        ((8001,), (15.0,), 1, 1, 1, 1, 3),
        ((8001,), (1.0,), 1, 1, 1, 1, 3),
        ((4001,), (1.0,), 1, 1, 1, 1, 3),
        ((50, 50, 50), (1.0, 1.0, 1.0), 1, 1, 1, 1, 3),
    ],
)
@pytest.mark.parametrize("boxtype", [PowerBox, LogNormalPowerBox])
def test_roundtrip_power_recovery(shape, size, boxtype, a, b, pkamp, low_slope, hi_slope) -> None:
    """Non-cubic Gaussian fields recover the input power over mixed odd/even shapes."""
    nrealizations = 20
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

    pmean = np.mean(power, axis=0)
    pstd = np.std(power, axis=0)
    expected = pkfunc(psobj.bin_centres)
    mask = np.isfinite(pstd[1:]) & (pstd[1:] > 0)
    zscore = np.abs(pmean[1:][mask] - expected[1:][mask]) / (
        pstd[1:][mask] / np.sqrt(nrealizations)
    )
    _assert_reasonable_power_recovery(zscore)
