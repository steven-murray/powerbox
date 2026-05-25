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
jdft = importlib.import_module("powerbox.jax.dft")
jtools = importlib.import_module("powerbox.jax.tools")
ndft = importlib.import_module("powerbox.dft")


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


def test_jax_powerbox_jit_delta_x_matches_eager_delta_x() -> None:
    pb = jpb.PowerBox(
        (16, 18),
        dim=2,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=(4.0, 6.0),
        key=jax.random.key(21),
    )

    key = jax.random.key(22)
    eager = pb.delta_x(key=key)
    compiled = pb.jit_delta_x(key=key)

    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=1e-7, atol=1e-7)


def test_jax_lognormal_jit_delta_x_matches_eager_delta_x() -> None:
    pb = jpb.LogNormalPowerBox(
        (16, 18),
        dim=2,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=(4.0, 6.0),
        key=jax.random.key(23),
    )

    key = jax.random.key(24)
    eager = pb.delta_x(key=key)
    compiled = pb.jit_delta_x(key=key)

    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=1e-7, atol=1e-7)


def test_jax_jit_delta_x_requires_key_if_not_provided_anywhere() -> None:
    pb = jpb.PowerBox(
        (8, 10),
        dim=2,
        pk=lambda k: (1 + k) ** -2.0,
        boxlength=(4.0, 5.0),
    )

    with pytest.raises(ValueError, match="PRNG key"):
        pb.jit_delta_x()


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


def test_jax_powerbox_rejects_bad_tuple_lengths() -> None:
    with pytest.raises(ValueError, match="length 2"):
        jpb.PowerBox(
            (16,),
            dim=2,
            pk=lambda k: (1 + k) ** -2.0,
            boxlength=(4.0, 5.0),
            key=jax.random.key(14),
        )

    with pytest.raises(ValueError, match="length 2"):
        jpb.PowerBox(
            (16, 18),
            dim=2,
            pk=lambda k: (1 + k) ** -2.0,
            boxlength=(4.0,),
            key=jax.random.key(15),
        )


def test_jax_powerbox_delta_k_rejects_negative_power() -> None:
    pb = jpb.PowerBox(
        (8, 10),
        dim=2,
        pk=lambda k: -jnp.ones_like(k),
        boxlength=(4.0, 5.0),
        key=jax.random.key(16),
    )
    with pytest.raises(ValueError, match="negative values"):
        pb.delta_k()


def test_jax_dft_frequency_wrappers_are_callable() -> None:
    field = jnp.arange(8.0).reshape(2, 4)
    shifted = jpb.fftshift(field)
    unshifted = jpb.ifftshift(shifted)
    np.testing.assert_allclose(np.asarray(unshifted), np.asarray(field))

    k = jdft.fftfreq(8, d=0.5, b=1)
    kr = jdft.rfftfreq(8, d=0.5, b=1)
    assert np.asarray(k).shape == (8,)
    assert np.asarray(kr).shape == (5,)


def test_jax_irfft_supports_inferred_and_scalar_n() -> None:
    full = np.random.default_rng(0).normal(size=(6, 8))
    reduced = np.fft.rfftn(full)
    reduced = np.fft.fftshift(reduced, axes=(0,))

    rec_default, _ = jdft.irfft(reduced, axes=(0, 1), a=0, b=2 * np.pi)
    rec_x0, _ = jdft.irfft(reduced, axes=(0, 1), x0=(0.2, -0.1), a=0, b=2 * np.pi)

    assert np.asarray(rec_default).shape == full.shape
    assert np.asarray(rec_x0).shape == full.shape

    with pytest.raises(ValueError, match="inconsistent with the reduced spectrum shape"):
        jdft.irfft(reduced, axes=(0, 1), N=8, a=0, b=2 * np.pi)

    with pytest.raises(ValueError, match="same length"):
        jdft.irfft(reduced, axes=(0, 1), N=(6, 8, 10), a=0, b=2 * np.pi)


def test_jax_tooling_validation_branches() -> None:
    with pytest.raises(ValueError, match="n_bins \\+ 1"):
        jtools.PowerSpectrum(
            power=jnp.ones(3),
            bin_edges=jnp.array([0.0, 1.0, 2.0]),
            bin_centres=jnp.array([0.5, 1.5, 2.5]),
        )

    with pytest.raises(ValueError, match="strictly positive"):
        jtools._bin_centres_from_edges(jnp.array([0.0, 1.0, 2.0]), log_bins=True)

    with pytest.raises(ValueError, match="same shape as the averaged"):
        jtools._resolve_radial_weights(
            [jnp.arange(4.0), jnp.arange(5.0)],
            jnp.ones((4, 5)),
            jnp.ones((2, 2)),
            ignore_zero_mode=False,
        )

    with pytest.raises(ValueError, match="coords and weights must have the same shape"):
        jtools._get_binweights(
            jnp.ones((4, 5)),
            jnp.ones((4, 4)),
            bins=4,
            bins_upto_boxlen=True,
        )

    with pytest.raises(NotImplementedError, match="complex field"):
        jtools._field_variance(
            indx=jnp.array([1, 1, 2, 2]),
            field=jnp.array([1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]),
            average=jnp.array([1.5, 3.5]),
            weights=1.0,
            v1=jnp.array([2.0, 2.0]),
        )

    field = jnp.arange(16.0).reshape(4, 4)
    coords = [jnp.linspace(-1.0, 1.0, 4), jnp.linspace(-2.0, 2.0, 4)]
    with pytest.raises(NotImplementedError, match="interpolation-based averaging"):
        jtools.angular_average(field, coords, bins=4, interpolation_method="linear")
    with pytest.raises(ValueError, match=r"same length as field\.ndim"):
        jtools.angular_average(field, [coords[0]], bins=4)
    with pytest.raises(ValueError, match="same shape as the field"):
        jtools.angular_average(field, jnp.ones((3, 3)), bins=4)

    with pytest.raises(ValueError, match=r"between 1 and field\.ndim"):
        jtools.angular_average_nd(field, coords, bins=4, ndims_to_avg=0)
    with pytest.raises(NotImplementedError, match="interpolation-based averaging"):
        jtools.angular_average_nd(field, coords, bins=4, interpolation_method="linear")

    with pytest.raises(ValueError, match="kmag must be provided"):
        jtools.ignore_zero_absk(coords, None)


def test_jax_get_power_validation_branches() -> None:
    field = jnp.arange(16.0).reshape(4, 4)
    with pytest.raises(ValueError, match="same shape"):
        jpb.get_power(field, 2.0, deltax2=jnp.arange(9.0).reshape(3, 3))
    with pytest.raises(ValueError, match="res_ndim must be between"):
        jpb.get_power(field, 2.0, res_ndim=-1)
    with pytest.raises(ValueError, match="res_ndim must be between"):
        jpb.get_power(field, 2.0, res_ndim=3)

    result = jpb.get_power(field, 2.0, res_ndim=0)
    assert result.bin_edges.size == 0
    assert result.bin_centres.size == 0


def test_dft_default_length_branches_are_exercised() -> None:
    x = np.random.default_rng(1).normal(size=(6, 8))

    # fft default L path (L and Lk both omitted)
    f_default, _ = ndft.fft(x, axes=(0, 1), a=0, b=2 * np.pi, nthreads=1)
    assert f_default.shape == x.shape

    # fft scalar Lk path
    f_lk, _ = ndft.fft(x, Lk=3.0, axes=(0, 1), a=0, b=2 * np.pi, nthreads=1)
    assert f_lk.shape == x.shape

    # ifft default Lk path
    x_ifft, _ = ndft.ifft(f_default, axes=(0, 1), a=0, b=2 * np.pi, nthreads=1)
    assert x_ifft.shape == x.shape

    # irfft scalar Lk path
    reduced = np.fft.rfftn(x)
    reduced = np.fft.fftshift(reduced, axes=(0,))
    x_irfft, _ = ndft.irfft(reduced, Lk=3.0, axes=(0, 1), N=x.shape, a=0, b=2 * np.pi, nthreads=1)
    assert x_irfft.shape == x.shape


def test_jax_tools_additional_branch_coverage() -> None:
    # PowerSpectrum validation for secondary optional arrays.
    with pytest.raises(ValueError, match="bin_avg must have length"):
        jtools.PowerSpectrum(
            power=jnp.ones(2),
            bin_edges=jnp.array([0.0, 1.0, 2.0]),
            bin_centres=jnp.array([0.5, 1.5]),
            bin_avg=jnp.ones(3),
        )

    with pytest.raises(ValueError, match="nsamples must have length"):
        jtools.PowerSpectrum(
            power=jnp.ones(2),
            bin_edges=jnp.array([0.0, 1.0, 2.0]),
            bin_centres=jnp.array([0.5, 1.5]),
            nsamples=jnp.ones(3),
        )

    with pytest.raises(ValueError, match="variance must have first dimension"):
        jtools.PowerSpectrum(
            power=jnp.ones(2),
            bin_edges=jnp.array([0.0, 1.0, 2.0]),
            bin_centres=jnp.array([0.5, 1.5]),
            variance=jnp.ones((3,)),
        )

    # _getbins log branch and _resolve_bins_upto_boxlen warning branch.
    coords = jnp.abs(jnp.arange(1, 10, dtype=float)).reshape(3, 3)
    bins = jtools._getbins(4, coords, log=True, bins_upto_boxlen=True)
    assert bins.shape == (5,)
    with pytest.warns(FutureWarning, match="In the future"):
        assert jtools._resolve_bins_upto_boxlen(4, None) is False

    # Complex averaging path and scalar-weights variance path.
    indx = jnp.array([1, 1, 2, 2])
    complex_avg = jtools._field_average(
        indx, jnp.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j]), 1.0, jnp.array([2.0, 2.0])
    )
    assert complex_avg.dtype.kind == "c"
    var = jtools._field_variance(
        indx,
        jnp.array([1.0, 3.0, 5.0, 7.0]),
        jnp.array([2.0, 6.0]),
        1.0,
        jnp.array([2.0, 2.0]),
    )
    assert np.all(np.isfinite(np.asarray(var)))

    # angular_average_nd branch where full-rank weights are sliced by leading index.
    field = jnp.arange(24.0).reshape(3, 4, 2)
    weights = jnp.ones_like(field)
    coords_nd = [jnp.linspace(-1.0, 1.0, 3), jnp.linspace(-1.5, 1.5, 4)]
    with pytest.warns(UserWarning, match="One or more radial bins had no cells within it."):
        out, outbins, outvar, outwght = jtools.angular_average_nd(
            field, coords_nd, bins=3, ndims_to_avg=2, weights=weights, bins_upto_boxlen=True
        )
    assert out.shape[1:] == (2,)
    assert outbins.shape == out.shape
    assert outvar is None
    assert outwght.shape == out.shape
