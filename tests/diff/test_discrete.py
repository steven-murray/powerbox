import pytest

import jax.numpy as jnp
from functools import partial

from powerbox.diff import LogNormalPowerBox, PowerBox, get_power

get_power = partial(get_power, bins_upto_boxlen=True)


def test_discrete_power_gaussian():
    pb = PowerBox(
        N=512,
        dim=2,
        boxlength=100.0,
        pk=lambda u: 0.1 * u**-1.5,
        ensure_physical=True,
    )

    box = pb.delta_x()

    sample = pb.create_discrete_sample(nbar=1000.0, delta_x=box)
    power, bins = get_power(sample, pb.boxlength, N=pb.N)

    res = jnp.mean(jnp.abs(power[50:-50] / (0.1 * bins[50:-50] ** -1.5) - 1))

    assert res < 1e-1

    # This re-grids the discrete sample into a box, basically to verify the
    # indexing used by meshgrid within `create_discrete_sample`.
    N = [pb.N] * pb.dim
    L = [pb.boxlength] * pb.dim
    edges = [jnp.linspace(-_L / 2.0, _L / 2.0, _n + 1) for _L, _n in zip(L, N)]
    delta_samp = jnp.histogramdd(sample, bins=edges, weights=None)[0].astype("float")

    # Check cross spectrum and assert a strong correlation
    cross, bins = get_power(delta_samp, pb.boxlength, deltax2=box)
    p2, bins = get_power(box, pb.boxlength)
    mask = (power > 0) & (p2 > 0)
    corr = cross[mask] / jnp.sqrt(power[mask]) / jnp.sqrt(p2[mask])
    corr_bar = jnp.mean(corr[jnp.isfinite(corr)])
    assert corr_bar > 10


def test_discrete_power_lognormal():
    pb = LogNormalPowerBox(
        N=512,
        dim=2,
        boxlength=100.0,
        pk=lambda u: 0.1 * u**-1.5,
        ensure_physical=True,
        seed=1212,
    )

    sample = pb.create_discrete_sample(nbar=1000.0)
    power, bins = get_power(sample, pb.boxlength, N=pb.N)

    res = jnp.mean(jnp.abs(power[50:-50] / (0.1 * bins[50:-50] ** -1.5) - 1))

    assert res < 1e-1

    with pytest.raises(ValueError):
        power, bins = get_power(sample.T, pb.boxlength, N=pb.N)

    with pytest.raises(ValueError):
        power, bins = get_power(sample.T, pb.boxlength, N=pb.N, deltax2=sample.T)

    with pytest.raises(ValueError):
        power, bins = get_power(sample, pb.boxlength, N=pb.N, deltax2=sample.T)

    get_power(sample, pb.boxlength, N=pb.N, deltax2=sample, dimensionless=False)


if __name__ == "__main__":
    test_discrete_power_gaussian()
    test_discrete_power_lognormal()
