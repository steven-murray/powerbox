import jax.numpy as jnp
import warnings
from functools import partial

from powerbox.diff import (
    PowerBox,
    get_power,
    ignore_zero_absk,
    ignore_zero_ki,
    power2delta,
)

get_power = partial(get_power, bins_upto_boxlen=True)


def test_power1d():
    p = [0] * 40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, a=0, b=1)

        p[i], k = get_power(pb.delta_x(), pb.boxlength, a=0, b=1)

    assert jnp.allclose(
        jnp.mean(jnp.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -2.0, rtol=2
    )


def test_power1d_n3():
    p = [0] * 40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0 * k**-3.0, boxlength=1.0, b=1)
        p[i], k = get_power(pb.delta_x(), pb.boxlength, b=1)

    assert jnp.allclose(
        jnp.mean(jnp.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -3.0, rtol=2
    )


def test_power1d_bigL():
    p = [0] * 40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0 * k**-3.0, boxlength=10.0, b=1)
        p[i], k = get_power(pb.delta_x(), pb.boxlength, b=1)

    assert jnp.allclose(
        jnp.mean(jnp.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -3.0, rtol=2
    )


def test_power1d_ordinary_freq():
    p = [0] * 40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0 * k**-3.0, boxlength=1.0)
        p[i], k = get_power(pb.delta_x(), pb.boxlength)

    assert jnp.allclose(
        jnp.mean(jnp.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -3.0, rtol=2
    )


def test_power1d_halfN():
    p = [0] * 40
    for i in range(40):
        pb = PowerBox(4001, dim=1, pk=lambda k: 1.0 * k**-3.0, boxlength=1.0, b=1)
        p[i], k = get_power(pb.delta_x(), pb.boxlength, b=1)

    assert jnp.allclose(
        jnp.mean(jnp.array(p), axis=0)[1000:], 1.0 * k[1000:] ** -3.0, rtol=2
    )


def test_power2d():
    p = [0] * 5
    for i in range(5):
        pb = PowerBox(200, dim=2, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
        p[i], k = get_power(pb.delta_x(), pb.boxlength, b=1)

    assert jnp.allclose(
        jnp.mean(jnp.array(p), axis=0)[100:], 1.0 * k[100:] ** -2.0, rtol=2
    )


def test_power3d():
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    p, k = get_power(pb.delta_x(), pb.boxlength, b=1)

    print(p / (1.0 * k**-2.0))
    assert jnp.allclose(p, 1.0 * k**-2.0, rtol=2)


def test_k_zero_ignore():
    pb = PowerBox(50, dim=2, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)

    dx = pb.delta_x()
    p1, k1 = get_power(dx, pb.boxlength, bin_ave=False)
    p0, k0 = get_power(dx, pb.boxlength, ignore_zero_mode=True, bin_ave=False)

    assert jnp.all(k1 == k0)

    assert jnp.all(p1[1:] == p0[1:])

    assert p1[0] != p0[0]


def test_k_weights():
    pb = PowerBox(50, dim=2, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)

    dx = pb.delta_x()

    k_weights = jnp.ones_like(dx)
    k_weights[:, 25] = 0

    p1, k1 = get_power(dx, pb.boxlength, bin_ave=False)
    p0, k0 = get_power(dx, pb.boxlength, bin_ave=False, k_weights=k_weights)

    assert jnp.all(k1 == k0)
    assert not jnp.allclose(p1, p0)


def test_prefactor_fnc():
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    pdelta, kdelta = get_power(pb.delta_x(), pb.boxlength, prefactor_fnc=power2delta)
    p, k = get_power(pb.delta_x(), pb.boxlength)

    assert jnp.all(k == kdelta)
    assert jnp.any(p != pdelta)


def test_k_weights_fnc():
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings(
            "ignore", message="One or more radial bins had no cells within it"
        )
        p_ki0, k_ki0 = get_power(pb.delta_x(), pb.boxlength, k_weights=ignore_zero_ki)
    p, k = get_power(pb.delta_x(), pb.boxlength, k_weights=ignore_zero_absk)

    assert not jnp.allclose(p, p_ki0)
