"""Tests for the JAX-backed ``powerbox`` namespace."""

from __future__ import annotations

import importlib
import itertools
from contextlib import nullcontext

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
jpb = importlib.import_module("powerbox.jax")


def _assert_full_hermitian(arr: np.ndarray) -> None:
    """Assert that a centred Fourier array is Hermitian."""
    unshifted = np.fft.ifftshift(arr)
    shape = unshifted.shape

    for index in itertools.product(*[range(axis_n) for axis_n in shape]):
        partner = tuple((-index[axis]) % shape[axis] for axis in range(len(shape)))
        if partner == index:
            assert np.allclose(unshifted[index].imag, 0, atol=1e-10)
        else:
            assert np.allclose(unshifted[index], np.conj(unshifted[partner]), atol=1e-10)


def _assert_reasonable_power_recovery(zscore: np.ndarray) -> None:
    """Require that power-recovery z-scores stay near 3-sigma overall."""
    frac_above_three_sigma = np.count_nonzero(zscore > 3.0) / zscore.size
    assert frac_above_three_sigma <= 0.1, zscore
    assert np.max(zscore) < 5.0, zscore


def test_jax_scalar_inputs_expand_to_tuple_geometry() -> None:
    pb = jpb.PowerBox(
        16,
        dim=2,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=4.0,
        key=jax.random.key(0),
    )

    assert pb.N == (16, 16)
    assert pb.boxlength == (4.0, 4.0)
    assert pb.x[0].shape == (16,)
    assert pb.x[1].shape == (16,)
    assert pb.kvec[0].shape == (16,)
    assert pb.kvec[1].shape == (9,)
    assert pb.delta_x().shape == (16, 16)


def test_jax_tuple_geometry_public_attributes_for_lognormal() -> None:
    pb = jpb.LogNormalPowerBox(
        (16, 18),
        dim=2,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=(4.0, 6.0),
        key=jax.random.key(1),
    )

    assert pb.N == (16, 18)
    assert pb.boxlength == (4.0, 6.0)
    assert pb.x[0].shape == (16,)
    assert pb.x[1].shape == (18,)
    assert pb.kvec[0].shape == (16,)
    assert pb.kvec[1].shape == (10,)
    np.testing.assert_array_less(-1 - 1e-12, np.asarray(pb.delta_x()))


def test_jax_lognormal_correlation_array_matches_irfft_of_power() -> None:
    pb = jpb.LogNormalPowerBox(
        (16, 18),
        dim=2,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=(4.0, 6.0),
        key=jax.random.key(9),
    )

    expected = pb._irfft_to_field(pb.power_array(), scale=pb.V)
    np.testing.assert_allclose(np.asarray(pb.correlation_array()), np.asarray(expected))


def test_jax_lognormal_delta_k_matches_gaussian_power_times_modes() -> None:
    pb = jpb.LogNormalPowerBox(
        (16, 18),
        dim=2,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=(4.0, 6.0),
        key=jax.random.key(10),
    )

    key = jax.random.key(11)
    expected = jnp.sqrt(pb.gaussian_power_array()) * pb.gauss_hermitian(key=key)
    np.testing.assert_allclose(np.asarray(pb.delta_k(key=key)), np.asarray(expected))


def test_jax_lognormal_delta_x_matches_manual_gaussianized_construction() -> None:
    pb = jpb.LogNormalPowerBox(
        (16, 18),
        dim=2,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=(4.0, 6.0),
        key=jax.random.key(12),
    )

    key = jax.random.key(13)
    dk = jnp.sqrt(pb.gaussian_power_array()) * pb._gaussian_modes_rfft(key=key)
    gaussian_field = pb._irfft_to_field(dk, scale=jnp.sqrt(pb.V))
    expected = jnp.exp(gaussian_field - jnp.var(gaussian_field) / 2) - 1
    np.testing.assert_allclose(np.asarray(pb.delta_x(key=key)), np.asarray(expected))


@pytest.mark.parametrize("shape", [(24, 31), (25, 30)])
def test_jax_get_power_recovers_input_for_non_cubic_boxes(shape: tuple[int, int]) -> None:
    def pkfunc(k):
        return (1 + k) ** -2

    boxlength = (4.0, 7.0)
    power = []
    nrealizations = 8

    for seed in range(nrealizations):
        pb = jpb.PowerBox(
            shape,
            dim=2,
            pk=pkfunc,
            boxlength=boxlength,
            key=jax.random.key(seed),
        )
        result = jpb.get_power(jpb.fftshift(pb.delta_x()), pb.boxlength, bins_upto_boxlen=True)
        power.append(np.asarray(result.power))

    pmean = np.mean(power, axis=0)
    pstd = np.std(power, axis=0)
    expected = pkfunc(np.asarray(result.bin_centres))
    mask = np.isfinite(pstd[1:]) & (pstd[1:] > 0)
    zscore = np.abs(pmean[1:][mask] - expected[1:][mask]) / (
        pstd[1:][mask] / np.sqrt(nrealizations)
    )
    _assert_reasonable_power_recovery(zscore)


@pytest.mark.parametrize("shape", [(4,), (5,), (4, 5), (5, 4), (4, 5, 6), (5, 4, 7)])
def test_jax_reduced_gaussian_modes_preserve_real_self_conjugate_modes(
    shape: tuple[int, ...],
) -> None:
    pb = jpb.PowerBox(
        shape,
        dim=len(shape),
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=tuple(float(axis + 2) for axis in range(len(shape))),
        key=jax.random.key(7),
    )
    gh = np.asarray(pb._gaussian_modes_rfft())

    assert gh.shape == (*shape[:-1], shape[-1] // 2 + 1)

    index_sets = []
    for axis, axis_n in enumerate(shape):
        values = [0] if axis == len(shape) - 1 else [axis_n // 2]
        if axis_n % 2 == 0:
            values.append(axis_n // 2 if axis == len(shape) - 1 else 0)
        index_sets.append(tuple(dict.fromkeys(values)))

    for idx in np.ndindex(*(len(values) for values in index_sets)):
        mode_index = tuple(index_sets[axis][i] for axis, i in enumerate(idx))
        assert np.allclose(gh[mode_index].imag, 0, atol=1e-10)


@pytest.mark.parametrize("shape", [(4,), (5,), (4, 5), (5, 4), (4, 5, 6), (5, 4, 7)])
def test_jax_reduced_gaussian_modes_boundary_surfaces_are_hermitian(shape: tuple[int, ...]) -> None:
    pb = jpb.PowerBox(
        shape,
        dim=len(shape),
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=tuple(float(axis + 2) for axis in range(len(shape))),
        key=jax.random.key(8),
    )
    gh = np.asarray(pb._gaussian_modes_rfft())
    surface_indices = [0]
    if shape[-1] % 2 == 0:
        surface_indices.append(shape[-1] // 2)

    for surface_index in surface_indices:
        surface = gh[..., surface_index]
        if np.ndim(surface) == 0:
            assert np.allclose(np.imag(surface), 0, atol=1e-10)
        else:
            _assert_full_hermitian(surface)


def test_jax_get_power_partial_average_retains_unbinned_axes() -> None:
    pb = jpb.PowerBox(
        (18, 20, 22),
        dim=3,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=(4.0, 5.0, 6.0),
        key=jax.random.key(2),
        b=1,
    )

    result = jpb.get_power(
        pb.delta_x(),
        pb.boxlength,
        b=1,
        res_ndim=2,
        prefactor_fnc=jpb.power2delta,
        get_variance=True,
        bins_upto_boxlen=True,
    )

    assert result.power.shape[1] == 22
    assert result.variance is not None
    assert result.k_unbinned is not None
    assert len(result.k_unbinned) == 1
    assert result.k_unbinned[0].shape == (22,)


@pytest.mark.parametrize("k_weights", [jpb.ignore_zero_absk, jpb.ignore_zero_ki])
def test_jax_get_power_accepts_boolean_k_weights_helpers(k_weights) -> None:
    pb = jpb.PowerBox(
        (18, 20, 22),
        dim=3,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=(4.0, 5.0, 6.0),
        key=jax.random.key(4),
        b=1,
    )

    expectation = (
        pytest.warns(UserWarning, match="One or more radial bins had no cells within it.")
        if k_weights is jpb.ignore_zero_ki
        else nullcontext()
    )

    with expectation:
        result = jpb.get_power(
            pb.delta_x(),
            pb.boxlength,
            b=1,
            k_weights=k_weights,
            bins_upto_boxlen=True,
        )

    assert result.power.ndim == 1
    assert result.nsamples is not None
    assert np.all(np.isfinite(np.asarray(result.power[1:])))


def test_jax_get_power_partial_average_accepts_ignore_zero_ki() -> None:
    pb = jpb.PowerBox(
        (18, 20, 22),
        dim=3,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=(4.0, 5.0, 6.0),
        key=jax.random.key(5),
        b=1,
    )

    with pytest.warns(UserWarning, match="One or more radial bins had no cells within it."):
        result = jpb.get_power(
            pb.delta_x(),
            pb.boxlength,
            b=1,
            res_ndim=2,
            k_weights=jpb.ignore_zero_ki,
            bins_upto_boxlen=True,
        )

    assert result.power.shape[1] == 22
    assert result.nsamples is not None
    assert result.nsamples.ndim == 1


def test_jax_fft_roundtrip_matches_input() -> None:
    field = jnp.arange(16.0).reshape(4, 4)
    transformed, _ = jpb.fft(field, L=2.0, a=0, b=2 * np.pi)
    recovered, _ = jpb.ifft(transformed, L=2.0, a=0, b=2 * np.pi)

    np.testing.assert_allclose(np.asarray(recovered), np.asarray(field))


def test_jax_unsupported_paths_raise() -> None:
    pb = jpb.PowerBox(
        16,
        dim=2,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=4.0,
        key=jax.random.key(3),
    )
    field = pb.delta_x()

    with pytest.raises(NotImplementedError, match="interpolation"):
        jpb.get_power(field, pb.boxlength, interpolation_method="linear")

    with pytest.raises(NotImplementedError, match="discrete samples"):
        jpb.get_power(field, pb.boxlength, N=pb.N)

    with pytest.raises(NotImplementedError, match="create_discrete_sample"):
        pb.create_discrete_sample(1.0)


def test_jax_rejects_non_integer_scalar_n() -> None:
    with pytest.raises(TypeError, match="integers"):
        jpb.PowerBox(
            16.5,
            dim=2,
            pk=lambda k: (1 + k) ** -2.0,
            boxlength=4.0,
            key=jax.random.key(6),
        )
