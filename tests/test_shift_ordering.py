"""Regression tests for shifted Fourier-space ordering."""

from __future__ import annotations

import numpy as np
import pytest

from powerbox import LogNormalPowerBox, PowerBox, dft

FOURIER_CONVENTIONS = [
    pytest.param(1.0, 1.0, id="cosmology-convention"),
    pytest.param(0.0, 2 * np.pi, id="numpy-convention"),
]


def _expected_shifted_ifft(box: PowerBox, field: np.ndarray, scale: float = 1.0) -> np.ndarray:
    n = field.shape[0]
    if n % 2 == 0:
        return (
            scale
            * dft.ifft(
                field,
                L=box.boxlength,
                a=box.fourier_a,
                b=box.fourier_b,
                backend=box.fftbackend,
            )[0]
        ).real

    dx = box.boxlength / n
    phase_1d = np.exp(-1j * box.fourier_b * box.kvec * dx / 2)
    phase = phase_1d
    for _ in range(1, field.ndim):
        phase = np.multiply.outer(phase, phase_1d)

    lk = 2 * np.pi / (dx * box.fourier_b)
    normalisation = (
        np.sqrt(np.abs(box.fourier_b) / (2 * np.pi) ** (1 + box.fourier_a)) ** field.ndim
    )
    transformed = (
        lk**field.ndim
        * box.fftbackend.ifftn(box.fftbackend.ifftshift(field * phase))
        * normalisation
    )
    return scale * box.fftbackend.fftshift(transformed).real


def _expected_powerbox_delta_x(box: PowerBox) -> np.ndarray:
    return _expected_shifted_ifft(box, box.delta_k(), scale=box.V)


def _expected_lognormal_correlation(box: LogNormalPowerBox) -> np.ndarray:
    return _expected_shifted_ifft(box, box.power_array(), scale=box.V)


def _expected_lognormal_delta_x(box: LogNormalPowerBox) -> np.ndarray:
    gaussian = _expected_shifted_ifft(box, box.delta_k(), scale=np.sqrt(box.V))
    return np.exp(gaussian - np.var(gaussian) / 2) - 1


@pytest.mark.parametrize("n", [32, 33])
@pytest.mark.parametrize(("a", "b"), FOURIER_CONVENTIONS)
def test_powerbox_delta_x_unshifts_delta_k(n: int, a: float, b: float) -> None:
    expected_box = PowerBox(
        n,
        lambda k: 1.0 / (1.0 + k**2),
        dim=2,
        boxlength=3.0,
        seed=1234,
        a=a,
        b=b,
    )
    actual_box = PowerBox(
        n,
        lambda k: 1.0 / (1.0 + k**2),
        dim=2,
        boxlength=3.0,
        seed=1234,
        a=a,
        b=b,
    )

    np.testing.assert_allclose(actual_box.delta_x(), _expected_powerbox_delta_x(expected_box))


@pytest.mark.parametrize("n", [32, 33])
@pytest.mark.parametrize(("a", "b"), FOURIER_CONVENTIONS)
def test_lognormal_correlation_array_unshifts_power_array(n: int, a: float, b: float) -> None:
    box = LogNormalPowerBox(
        n,
        lambda k: 0.01 / (1.0 + k**2),
        dim=2,
        boxlength=3.0,
        seed=1234,
        a=a,
        b=b,
    )

    np.testing.assert_allclose(box.correlation_array(), _expected_lognormal_correlation(box))


@pytest.mark.parametrize("n", [32, 33])
@pytest.mark.parametrize(("a", "b"), FOURIER_CONVENTIONS)
def test_lognormal_delta_x_unshifts_delta_k(n: int, a: float, b: float) -> None:
    expected_box = LogNormalPowerBox(
        n,
        lambda k: 0.01 / (1.0 + k**2),
        dim=2,
        boxlength=3.0,
        seed=1234,
        a=a,
        b=b,
    )
    actual_box = LogNormalPowerBox(
        n,
        lambda k: 0.01 / (1.0 + k**2),
        dim=2,
        boxlength=3.0,
        seed=1234,
        a=a,
        b=b,
    )

    np.testing.assert_allclose(actual_box.delta_x(), _expected_lognormal_delta_x(expected_box))
