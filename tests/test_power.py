import pytest
import numpy as np
import warnings
from functools import partial

from powerbox import PowerBox, PowerSpectrum, get_power, ignore_zero_absk, ignore_zero_ki, power2delta

get_power = partial(get_power, bins_upto_boxlen=True)


def test_power1d():
    p = [0] * 40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, a=0, b=1)
        res = get_power(pb.delta_x(), pb.boxlength, a=0, b=1)
        p[i] = res.power
        k = res.bin_avg  # same bin structure for every iteration

    np.testing.assert_allclose(
        np.mean(np.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -2.0, rtol=2
    )


def test_power1d_n3():
    p = [0] * 40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0 * k**-3.0, boxlength=1.0, b=1)
        res = get_power(pb.delta_x(), pb.boxlength, b=1)
        p[i] = res.power
        k = res.bin_avg  # same bin structure for every iteration

    np.testing.assert_allclose(
        np.mean(np.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -3.0, rtol=2
    )


def test_power1d_bigL():
    p = [0] * 40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0 * k**-3.0, boxlength=10.0, b=1)
        res = get_power(pb.delta_x(), pb.boxlength, b=1)
        p[i] = res.power
        k = res.bin_avg  # same bin structure for every iteration

    np.testing.assert_allclose(
        np.mean(np.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -3.0, rtol=2
    )


def test_power1d_ordinary_freq():
    p = [0] * 40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0 * k**-3.0, boxlength=1.0)
        res = get_power(pb.delta_x(), pb.boxlength)
        p[i] = res.power
        k = res.bin_avg  # same bin structure for every iteration

    np.testing.assert_allclose(
        np.mean(np.array(p), axis=0)[2000:], 1.0 * k[2000:] ** -3.0, rtol=2
    )


def test_power1d_halfN():
    p = [0] * 40
    for i in range(40):
        pb = PowerBox(4001, dim=1, pk=lambda k: 1.0 * k**-3.0, boxlength=1.0, b=1)
        res = get_power(pb.delta_x(), pb.boxlength, b=1)
        p[i] = res.power
        k = res.bin_avg  # same bin structure for every iteration

    np.testing.assert_allclose(
        np.mean(np.array(p), axis=0)[1000:], 1.0 * k[1000:] ** -3.0, rtol=2
    )


def test_power2d():
    p = [0] * 5
    for i in range(5):
        pb = PowerBox(200, dim=2, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
        res = get_power(pb.delta_x(), pb.boxlength, b=1)
        p[i] = res.power
        k = res.bin_avg  # same bin structure for every iteration

    np.testing.assert_allclose(
        np.mean(np.array(p), axis=0)[100:], 1.0 * k[100:] ** -2.0, rtol=2
    )


def test_power3d():
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    res = get_power(pb.delta_x(), pb.boxlength, b=1)
    p = res.power
    k = res.bin_avg

    print(p / (1.0 * k**-2.0))
    np.testing.assert_allclose(p, 1.0 * k**-2.0, rtol=2)


def test_k_zero_ignore():
    pb = PowerBox(50, dim=2, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)

    dx = pb.delta_x()
    res1 = get_power(dx, pb.boxlength)
    res0 = get_power(dx, pb.boxlength, ignore_zero_mode=True)

    # bin_edges and bin_centres are structurally identical; bin_avg differs
    # because the k=0 mode is excluded from the weighted average in bin 0.
    np.testing.assert_allclose(res1.bin_edges, res0.bin_edges)

    np.testing.assert_allclose(res1.power[1:], res0.power[1:])

    assert res1.power[0] != res0.power[0]


def test_k_weights():
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)

    dx = pb.delta_x()

    k_weights = np.ones_like(dx)
    k_weights[:, 25] = 0

    res1 = get_power(dx, pb.boxlength)
    res0 = get_power(dx, pb.boxlength, k_weights=k_weights)

    np.testing.assert_allclose(res1.bin_edges, res0.bin_edges)
    assert not np.allclose(res1.power, res0.power)

    k_space_field = dx + dx * 1j
    # large scale modes removed
    k_space_field[15:35, 15:35, 15:35] = 0
    real_space_field = np.fft.ifftn(np.fft.fftshift(k_space_field)).real
    res3 = get_power(real_space_field, pb.boxlength)
    k_space_field = dx + dx * 1j
    real_space_field = np.fft.ifftn(np.fft.fftshift(k_space_field)).real
    # mask out the low-k modes
    k_weights = np.ones_like(dx)
    k_weights[15:35, 15:35, 15:35] = 0
    # set the masked region to zero in the full box
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings(
            "ignore", message="One or more radial bins had no cells within it"
        )
        res2 = get_power(
            real_space_field, pb.boxlength, k_weights=k_weights, bins=res3.bin_edges
        )
    # we expect that the PS of the small box is similar to the PS
    # of the big box with the low-k modes removed

    assert np.all(res3.bin_edges == res2.bin_edges)
    assert np.allclose(res2.power[~np.isnan(res2.power)], res3.power[~np.isnan(res2.power)])


def test_prefactor_fnc():
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    res_delta = get_power(pb.delta_x(), pb.boxlength, prefactor_fnc=power2delta)
    res = get_power(pb.delta_x(), pb.boxlength)

    np.testing.assert_allclose(res.bin_avg, res_delta.bin_avg)
    assert np.any(res.power != res_delta.power)


def test_k_weights_fnc():
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings(
            "ignore", message="One or more radial bins had no cells within it"
        )
        res_ki0 = get_power(pb.delta_x(), pb.boxlength, k_weights=ignore_zero_ki)
    res = get_power(pb.delta_x(), pb.boxlength, k_weights=ignore_zero_absk)

    assert not np.allclose(res.power, res_ki0.power)


def test_res_ndim_zero():
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    result = get_power(pb.delta_x(), pb.boxlength, res_ndim=0)

    assert result.power.ndim == 3
    assert len(result.bin_edges) == 0
    assert len(result.bin_centres) == 0
    assert result.bin_avg is None
    assert result.nsamples is None
    assert result.variance is None
    assert len(result.k_unbinned) == 3


def test_res_ndim_invalid():
    pb = PowerBox(50, dim=3, pk=lambda k: 1.0 * k**-2.0, boxlength=1.0, b=1)
    with pytest.raises(ValueError, match="res_ndim must be between"):
        get_power(pb.delta_x(), pb.boxlength, res_ndim=-1)
    with pytest.raises(ValueError, match="res_ndim must be between"):
        get_power(pb.delta_x(), pb.boxlength, res_ndim=4)


def test_power_spectrum_is_powerspectrum():
    """get_power returns a PowerSpectrum instance."""
    pb = PowerBox(50, dim=2, pk=lambda k: k**-2.0, boxlength=1.0, b=1)
    result = get_power(pb.delta_x(), pb.boxlength, b=1)
    assert isinstance(result, PowerSpectrum)


def test_power_spectrum_attributes():
    """All expected attributes exist and have consistent shapes."""
    pb = PowerBox(50, dim=2, pk=lambda k: k**-2.0, boxlength=1.0, b=1)
    result = get_power(pb.delta_x(), pb.boxlength, b=1)

    n = len(result.power)
    assert result.bin_edges.shape == (n + 1,)
    assert result.bin_centres.shape == (n,)
    assert result.bin_avg.shape == (n,)
    assert result.nsamples.shape == (n,)
    assert result.variance is None
    assert result.k_unbinned is None


def test_power_spectrum_variance():
    """variance is populated when get_variance=True."""
    pb = PowerBox(50, dim=2, pk=lambda k: k**-2.0, boxlength=1.0, b=1)
    result = get_power(pb.delta_x(), pb.boxlength, b=1, get_variance=True)

    assert result.variance is not None
    assert result.variance.shape == result.bin_centres.shape


def test_power_spectrum_bin_edges_monotonic():
    """bin_edges should be monotonically increasing."""
    pb = PowerBox(50, dim=2, pk=lambda k: k**-2.0, boxlength=1.0, b=1)
    result = get_power(pb.delta_x(), pb.boxlength, b=1)
    assert np.all(np.diff(result.bin_edges) > 0)


def test_power_spectrum_bin_centres_in_edges():
    """bin_centres should fall inside corresponding bin_edges."""
    pb = PowerBox(50, dim=2, pk=lambda k: k**-2.0, boxlength=1.0, b=1)
    result = get_power(pb.delta_x(), pb.boxlength, b=1)
    assert np.all(result.bin_centres >= result.bin_edges[:-1])
    assert np.all(result.bin_centres <= result.bin_edges[1:])


def test_power_spectrum_log_bins():
    """log_bins=True produces log-spaced bin_edges and geometric bin_centres."""
    pb = PowerBox(50, dim=2, pk=lambda k: k**-2.0, boxlength=1.0, b=1)
    result = get_power(pb.delta_x(), pb.boxlength, b=1, log_bins=True)

    # bin_centres should be geometric means of adjacent edges
    expected_centres = np.exp(
        (np.log(result.bin_edges[1:]) + np.log(result.bin_edges[:-1])) / 2
    )
    np.testing.assert_allclose(result.bin_centres, expected_centres)


def test_power_spectrum_partial_average():
    """res_ndim < ndim fills k_unbinned and keeps power multi-dimensional."""
    pb = PowerBox(50, dim=3, pk=lambda k: k**-2.0, boxlength=1.0, b=1)
    result = get_power(pb.delta_x(), pb.boxlength, b=1, res_ndim=2)

    assert result.power.ndim == 2
    assert result.bin_avg.ndim == 1
    assert result.nsamples.ndim == 1
    assert result.k_unbinned is not None
    assert len(result.k_unbinned) == 1


def test_powerspectrum_validation_mismatch():
    """PowerSpectrum raises ValueError for inconsistent shapes."""
    power = np.ones(10)
    edges = np.linspace(0, 1, 12)  # 11 edges != 10+1
    centres = np.linspace(0, 1, 10)
    avg = np.linspace(0, 1, 10)
    nsamples = np.ones(10)

    with pytest.raises(ValueError, match="bin_edges must have length"):
        PowerSpectrum(
            power=power,
            bin_edges=edges,
            bin_centres=centres,
            bin_avg=avg,
            nsamples=nsamples,
        )


def test_powerspectrum_validation_bin_avg_shape():
    """PowerSpectrum raises ValueError when bin_avg has wrong length."""
    power = np.ones(10)
    edges = np.linspace(0, 1, 11)
    centres = (edges[1:] + edges[:-1]) / 2

    with pytest.raises(ValueError, match="bin_avg must have length"):
        PowerSpectrum(
            power=power,
            bin_edges=edges,
            bin_centres=centres,
            bin_avg=np.ones(5),  # wrong length
        )


def test_powerspectrum_validation_nsamples_shape():
    """PowerSpectrum raises ValueError when nsamples has wrong length."""
    power = np.ones(10)
    edges = np.linspace(0, 1, 11)
    centres = (edges[1:] + edges[:-1]) / 2

    with pytest.raises(ValueError, match="nsamples must have length"):
        PowerSpectrum(
            power=power,
            bin_edges=edges,
            bin_centres=centres,
            nsamples=np.ones(5),  # wrong length
        )


def test_powerspectrum_validation_variance_shape():
    """PowerSpectrum raises ValueError when variance has wrong shape."""
    power = np.ones(10)
    edges = np.linspace(0, 1, 11)
    centres = (edges[1:] + edges[:-1]) / 2
    avg = centres.copy()
    nsamples = np.ones(10)

    with pytest.raises(ValueError, match="variance must have first dimension"):
        PowerSpectrum(
            power=power,
            bin_edges=edges,
            bin_centres=centres,
            bin_avg=avg,
            nsamples=nsamples,
            variance=np.ones(5),
        )

