"""Test the powerbox interface.

Note that these tests are not so much about checking that the output is physically
correct, but more about checking that the interface behaves as expected, and that the
various options are doing what they are supposed to do.
"""

import warnings

import numpy as np
import pytest

from powerbox import NumpyFFT, PowerBox


def test_scalar_inputs_expand_to_tuple_geometry() -> None:
    """Scalar constructor inputs are normalized to tuple-valued reduced geometry."""
    # Both deprecated names are exercised here on purpose, so both warnings are expected.
    with pytest.warns(DeprecationWarning, match="parameter is deprecated") as record:
        pb = PowerBox(N=16, dim=2, pk=lambda k: (1 + k) ** -2.0, boxlength=4.0, seed=1234)

    messages = " ".join(str(warning.message) for warning in record)
    assert "`N` parameter is deprecated" in messages
    assert "`boxlength` parameter is deprecated" in messages

    assert pb.shape == (16, 16)
    assert pb.size == (4.0, 4.0)
    assert isinstance(pb.x, tuple)
    assert isinstance(pb.kvec, tuple)
    assert pb.x[0].shape == (16,)
    assert pb.x[1].shape == (16,)
    assert pb.kvec[0].shape == (16,)
    assert pb.kvec[1].shape == (9,)
    assert pb.delta_x().shape == (16, 16)


def test_tuple_inputs_expose_axis_aware_geometry() -> None:
    """Tuple inputs produce per-axis public geometry for non-cubic boxes."""
    pb = PowerBox(
        shape=(15, 18),
        pk=lambda k: (1 + k) ** -2.0,
        size=(3.0, 9.0),
        seed=1234,
    )

    assert pb.shape == (15, 18)
    assert pb.size == (3.0, 9.0)
    assert isinstance(pb.x, tuple)
    assert isinstance(pb.kvec, tuple)
    assert len(pb.x) == len(pb.kvec) == pb.dim
    assert pb.x[0].shape == (15,)
    assert pb.x[1].shape == (18,)
    assert pb.kvec[0].shape == (15,)
    assert pb.kvec[1].shape == (10,)
    assert pb.delta_x().shape == (15, 18)
    assert pb.r.shape == (15, 18)


@pytest.mark.parametrize(
    ("shape", "size", "error", "match"),
    [
        ((15, 18), (3.0,), ValueError, "size must have same length as dim"),
        ((15, 18), (3.0, "bad"), ValueError, "could not convert string to float"),
    ],
)
def test_tuple_input_validation(shape, size, error, match) -> None:
    """Tuple-valued geometry inputs validate length and element types."""
    with pytest.raises(error, match=match):
        PowerBox(shape=shape, pk=lambda k: (1 + k) ** -2.0, size=size)


def test_non_volume_normalized_powerbox_uses_input_power_directly() -> None:
    """Disabling volume normalization leaves the power callable unchanged."""
    pb = PowerBox(
        shape=(16, 16),
        pk=lambda k: k + 1.0,
        size=(4.0, 4.0),
        seed=1234,
        vol_normalised_power=False,
    )

    np.testing.assert_allclose(pb.pk(np.array([1.5, 2.5])), np.array([2.5, 3.5]))


def test_negative_power_raises() -> None:
    """Negative input power remains a hard error."""
    pb = PowerBox(shape=(16, 16), pk=lambda k: -np.ones_like(k), size=(4.0, 4.0), seed=1234)

    with pytest.raises(ValueError, match="returned negative values"):
        pb.delta_k()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "You must provide 'shape'"),
        ({"shape": (15, 18), "dim": 3}, "shape must have same length as dim"),
        ({"shape": (15, 0)}, "All elements of shape must be positive integers"),
        ({"shape": (15, -2)}, "All elements of shape must be positive integers"),
        ({"shape": (15, 18), "size": (3.0, 0.0)}, "All elements of size must be positive"),
        ({"shape": (15, 18), "size": (3.0, -1.0)}, "All elements of size must be positive"),
        # The deprecated aliases must not be combined with the names that replace them.
        ({"N": 15, "shape": (15, 18)}, "Don't provide both N and shape"),
        ({"shape": (15, 18), "boxlength": 3.0, "size": (3.0, 9.0)}, "Don't provide both boxlength"),
    ],
)
def test_geometry_validation_rejects_inconsistent_inputs(kwargs, match) -> None:
    """Contradictory or non-physical geometry is rejected at construction."""
    with warnings.catch_warnings():
        # The last two cases pass a deprecated name on purpose, to check it against its
        # replacement; the deprecation itself is asserted separately.
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match=match):
            PowerBox(pk=lambda k: (1 + k) ** -2.0, **kwargs)


def test_scalar_n_without_dim_defaults_to_two_dimensions() -> None:
    """The deprecated scalar ``N`` keeps its historical two-dimensional default."""
    with pytest.warns(DeprecationWarning, match="`N` parameter is deprecated"):
        pb = PowerBox(N=16, pk=lambda k: (1 + k) ** -2.0)

    assert pb.shape == (16, 16)
    assert pb.dim == 2


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("L", "size"), ("V", "volume"), ("Ntot", "total_ncells"), ("N", None)],
)
def test_deprecated_aliases_warn_and_agree_with_their_replacements(alias, canonical) -> None:
    """The v1.2 removal candidates still work, and still return the same thing."""
    pb = PowerBox(shape=(15, 18), pk=lambda k: (1 + k) ** -2.0, size=(3.0, 9.0))

    if canonical is None:
        # `N` is an input alias rather than a derived quantity, so it is simply unset here.
        assert pb.N is None
        return

    with pytest.warns(DeprecationWarning, match=f"`{alias}` attribute is deprecated"):
        assert getattr(pb, alias) == getattr(pb, canonical)


@pytest.mark.parametrize("nthreads", [0, 1, False, True])
def test_nthreads_that_mean_numpy_are_accepted(nthreads) -> None:
    """``nthreads`` of 0 or 1 (or a bool) selects NumPy's FFT, as documented."""
    pb = PowerBox(shape=(16, 16), pk=lambda k: (1 + k) ** -2.0, nthreads=nthreads, seed=1)

    assert isinstance(pb.fftbackend, NumpyFFT)
    assert pb.delta_x().shape == (16, 16)


def test_negative_nthreads_is_rejected() -> None:
    """Only a negative thread count is meaningless."""
    with pytest.raises(ValueError, match="'nthreads' must be >= 0"):
        PowerBox(shape=(16, 16), pk=lambda k: (1 + k) ** -2.0, nthreads=-1)


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [({"shape": 16}, "shape"), ({"shape": (16, 16), "size": 4.0}, "size")],
)
def test_bare_numbers_for_per_axis_geometry_get_a_helpful_error(kwargs, name) -> None:
    """``shape`` and ``size`` are sequences; a bare number says so rather than 'not iterable'."""
    with pytest.raises(TypeError, match=rf"`{name}` must be a sequence with one entry per axis"):
        PowerBox(pk=lambda k: (1 + k) ** -2.0, **kwargs)


def test_boxlength_attribute_is_a_deprecated_alias_of_size() -> None:
    """Reading ``boxlength`` keeps working for one more minor version, and warns."""
    pb = PowerBox(shape=(15, 18), pk=lambda k: (1 + k) ** -2.0, size=(3.0, 9.0))

    with pytest.warns(DeprecationWarning, match="`boxlength` attribute is deprecated") as record:
        assert pb.boxlength == pb.size

    assert "Use `size`" in str(record[0].message)


def test_deprecated_attribute_warnings_point_at_the_replacement_that_exists() -> None:
    """Every deprecation message names an attribute that actually exists."""
    pb = PowerBox(shape=(15, 18), pk=lambda k: (1 + k) ** -2.0, size=(3.0, 9.0))

    for alias in ("boxlength", "L", "V", "Ntot"):
        with pytest.warns(DeprecationWarning, match=f"`{alias}` attribute is deprecated") as record:
            getattr(pb, alias)
        replacement = str(record[0].message).split("Use `")[1].split("`")[0]
        assert hasattr(pb, replacement)
