"""Test the powerbox interface.

Note that these tests are not so much about checking that the output is physically
correct, but more about checking that the interface behaves as expected, and that the
various options are doing what they are supposed to do.
"""

import numpy as np
import pytest

from powerbox import PowerBox


def test_scalar_inputs_expand_to_tuple_geometry() -> None:
    """Scalar constructor inputs are normalized to tuple-valued reduced geometry."""
    pb = PowerBox(N=16, dim=2, pk=lambda k: (1 + k) ** -2.0, boxlength=4.0, seed=1234)

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
        boxlength=4.0,
        seed=1234,
        vol_normalised_power=False,
    )

    np.testing.assert_allclose(pb.pk(np.array([1.5, 2.5])), np.array([2.5, 3.5]))


def test_negative_power_raises() -> None:
    """Negative input power remains a hard error."""
    pb = PowerBox(shape=(16, 16), pk=lambda k: -np.ones_like(k), boxlength=4.0, seed=1234)

    with pytest.raises(ValueError, match="returned negative values"):
        pb.delta_k()
