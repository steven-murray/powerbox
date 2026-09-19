"""Canonical tests that the Fourier convention ``(a, b)`` is handled consistently.

The Fourier convention is a labelling choice, not physics: a given physical field can be
described in any convention, provided the input power spectrum is re-expressed
accordingly. These tests assert that identity deterministically, to machine precision.

Because convention handling is established here, the rest of the test suite is free to
use a single convention.
"""

import numpy as np
import pytest

from powerbox import LogNormalPowerBox, PowerBox

# The reference convention. Each test maps this onto some other convention and requires
# the two to produce exactly the same field.
A_REF, B_REF = 1.0, 1.0

OTHER_CONVENTIONS = [
    (0.0, 1.0),  # xi(0) is a factor (2*pi)^(dim/2) larger than the reference
    (0.0, 2 * np.pi),  # the numpy convention
    (1.0, 2 * np.pi),
    (0.5, 3.0),  # deliberately not a convention anyone uses
]


def synthesis_norm(a: float, b: float, dim: int) -> float:
    """Return the mode amplitude factor ``A`` of :attr:`PowerBox.synthesis_norm`."""
    return ((2 * np.pi) ** (1 - a) / b) ** (dim / 2)


def equivalent_pk(pk, a: float, b: float, dim: int):
    r"""Re-express ``pk`` from the reference convention into the convention ``(a, b)``.

    Convention ``b`` rescales the wavenumber grid as :math:`k \propto 1/b`, and the
    convention pre-factors rescale the mode amplitudes by
    :attr:`~powerbox.PowerBox.synthesis_norm`. Applying both to the input power spectrum
    describes the same physical field in the new convention.
    """
    amplitude = (synthesis_norm(A_REF, B_REF, dim) / synthesis_norm(a, b, dim)) ** 2

    def pk_transformed(k):
        return amplitude * pk(k * b / B_REF)

    return pk_transformed


def pk_reference(k):
    # A modest amplitude, so that the lognormal construction is valid in every
    # convention tested (see test_lognormal_rejects_invalid_power).
    return 0.01 / (k**2 + 1 / k)


@pytest.mark.parametrize(("a", "b"), OTHER_CONVENTIONS)
@pytest.mark.parametrize("boxtype", [PowerBox, LogNormalPowerBox])
@pytest.mark.parametrize("shape", [(24, 32), (25, 31)])
def test_field_is_identical_across_conventions(boxtype, a, b, shape) -> None:
    """The same physical field is realized in any Fourier convention."""
    size = (120.0, 180.0)
    dim = len(shape)

    reference = boxtype(shape=shape, pk=pk_reference, size=size, a=A_REF, b=B_REF, seed=3).delta_x()
    other = boxtype(
        shape=shape,
        pk=equivalent_pk(pk_reference, a, b, dim),
        size=size,
        a=a,
        b=b,
        seed=3,
    ).delta_x()

    np.testing.assert_allclose(other, reference, rtol=1e-10, atol=1e-12 * np.abs(reference).max())


@pytest.mark.parametrize(("a", "b"), [(A_REF, B_REF), *OTHER_CONVENTIONS])
def test_correlation_array_is_the_true_correlation_function(a, b) -> None:
    """``correlation_array()[0]`` is the variance of the equivalent Gaussian field.

    This is the invariant the lognormal transform depends on. It is easy to get wrong by a
    factor of :attr:`PowerBox.synthesis_norm`, which is one for both ``(1, 1)`` and
    ``(0, 2*pi)`` and so leaves the standard conventions looking correct.
    """
    shape, size = (49, 71), (120.0, 180.0)

    def pk(k):
        return 0.01 / (k**2 + 1 / k)

    lnpb = LogNormalPowerBox(shape=shape, pk=pk, size=size, a=a, b=b, seed=0)
    xi_zero = lnpb.correlation_array()[0, 0]

    variance = np.mean(
        [
            np.var(PowerBox(shape=shape, pk=pk, size=size, a=a, b=b, seed=seed).delta_x())
            for seed in range(16)
        ]
    )

    # 16 realizations of ~3500 independent modes; the scatter on the mean variance is
    # well under a percent.
    assert xi_zero == pytest.approx(variance, rel=0.02)


def test_lognormal_rejects_invalid_power() -> None:
    """A lognormal field that cannot exist on the grid raises, rather than being faked.

    ``log(1 + xi)`` is not positive semi-definite for large-variance fields, so there is
    no Gaussian field whose exponential has the requested power spectrum. Taking the
    absolute value of the required Gaussian power spectrum would hide that, and yield a
    plausible-looking field with the wrong power.
    """
    pb = LogNormalPowerBox(
        shape=(49, 71),
        pk=lambda k: 1.0 / (k**2 + 1 / k),
        size=(120.0, 180.0),
        a=0,
        b=1,
        seed=0,
    )
    # This choice gives xi(0) ~ 2.8, well into the invalid regime.
    assert pb.correlation_array()[0, 0] > 1

    with pytest.raises(ValueError, match="not a valid correlation function"):
        pb.delta_x()


def test_valid_lognormal_power_is_non_negative() -> None:
    """A valid lognormal configuration passes the positive-definiteness check."""
    pb = LogNormalPowerBox(shape=(49, 71), pk=pk_reference, size=(120.0, 180.0), a=1, b=1, seed=0)
    assert pb.correlation_array()[0, 0] < 1
    assert np.all(pb.gaussian_power_array() >= 0)
