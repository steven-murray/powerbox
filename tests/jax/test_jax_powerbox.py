"""Tests of the JAX implementation of PowerBox."""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
jpb = importlib.import_module("powerbox.jax")
jpb_powerbox = importlib.import_module("powerbox.jax.powerbox")
jdft = importlib.import_module("powerbox.jax.dft")
jtools = importlib.import_module("powerbox.jax.tools")
ndft = importlib.import_module("powerbox.dft")


def _assert_reasonable_power_recovery(zscore: np.ndarray) -> None:
    """Require that power-recovery z-scores stay near 3-sigma overall."""
    frac_above_three_sigma = np.count_nonzero(zscore > 3.0) / zscore.size
    assert frac_above_three_sigma <= 0.1, zscore
    assert np.max(zscore) < 5.0, zscore


@pytest.mark.parametrize("boxtype", [jpb.PowerBox, jpb.LogNormalPowerBox])
def test_jax_powerbox_jit_delta_x_matches_eager_delta_x(boxtype) -> None:
    pb = boxtype(
        shape=(16, 18),
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 6.0),
        key=jax.random.key(21),
        usejit=True,
    )

    key = jax.random.key(22)
    eager = pb._delta_x_eager(key=key)
    compiled = pb.delta_x(key=key)

    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=1e-7, atol=1e-7)


def test_jax_delta_x_requires_key_if_not_provided_anywhere() -> None:
    pb = jpb.PowerBox(
        shape=(8, 10),
        dim=2,
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 5.0),
    )

    with pytest.raises(ValueError, match="PRNG key"):
        pb.delta_x()


def test_jax_default_usejit_heuristic_can_be_forced_via_threshold(monkeypatch) -> None:
    monkeypatch.setattr(jpb_powerbox, "DEFAULT_JIT_NTOT_THRESHOLD", 10_000)
    eager_pb = jpb.PowerBox(
        shape=(8, 8),
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 4.0),
        key=jax.random.key(25),
    )
    assert eager_pb.usejit is False

    monkeypatch.setattr(jpb_powerbox, "DEFAULT_JIT_NTOT_THRESHOLD", 10)
    jit_pb = jpb.PowerBox(
        shape=(8, 8),
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 4.0),
        key=jax.random.key(26),
    )
    assert jit_pb.usejit is True


@pytest.mark.parametrize(
    ("threshold", "usejit", "expect_warning"),
    [
        # Above the threshold: eager was chosen by the heuristic, so say so, once.
        (10_000, None, True),
        # Eager was asked for explicitly: the user knows, so stay quiet.
        (10_000, False, False),
        # Below the threshold: JIT is the default, so there is nothing to report.
        (1, None, False),
        (1, True, False),
    ],
)
def test_jax_eager_mode_warns_only_when_chosen_implicitly(
    monkeypatch, threshold, usejit, expect_warning
) -> None:
    """The execution-policy warning fires only for an implicitly-chosen eager path."""
    monkeypatch.setattr(jpb_powerbox, "DEFAULT_JIT_NTOT_THRESHOLD", threshold)
    pb = jpb.PowerBox(
        shape=(8, 8),
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 4.0),
        key=jax.random.key(27),
        usejit=usejit,
    )
    assert pb.usejit is (threshold == 1)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        pb.delta_x()
        pb.delta_x()

    messages = [str(warning.message) for warning in record]
    policy_warnings = [m for m in messages if "chosen by the size heuristic" in m]

    # Warned at most once, however many times delta_x() is called.
    assert len(policy_warnings) == (1 if expect_warning else 0)
    assert messages == policy_warnings, f"unexpected warnings: {messages}"


def test_jax_lognormal_correlation_array_matches_irfft_of_power() -> None:
    pb = jpb.LogNormalPowerBox(
        shape=(16, 18),
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 6.0),
        key=jax.random.key(9),
    )

    expected = pb._irfft_to_field(pb.power_array(), scale=pb.volume)
    np.testing.assert_allclose(np.asarray(pb.correlation_array()), np.asarray(expected))


def test_jax_lognormal_delta_k_matches_gaussian_power_times_modes() -> None:
    pb = jpb.LogNormalPowerBox(
        shape=(16, 18),
        pk=lambda k: (1 + k) ** -2.0,
        size=(4.0, 6.0),
        key=jax.random.key(10),
    )

    key = jax.random.key(11)
    expected = jnp.sqrt(pb.gaussian_power_array()) * pb.gauss_hermitian(key=key)
    np.testing.assert_allclose(np.asarray(pb.delta_k(key=key)), np.asarray(expected))


@pytest.mark.parametrize("shape", [(24, 31), (25, 30)])
def test_jax_get_power_recovers_input_for_non_cubic_boxes(shape: tuple[int, int]) -> None:
    def pkfunc(k):
        return (1 + k) ** -2

    size = (4.0, 7.0)
    power = []
    nrealizations = 8

    for seed in range(nrealizations):
        pb = jpb.PowerBox(
            shape=shape,
            pk=pkfunc,
            size=size,
            key=jax.random.key(seed),
        )
        result = jpb.get_power(jpb.fftshift(pb.delta_x()), pb.size, bins_upto_boxlen=True)
        power.append(np.asarray(result.power))

    pmean = np.mean(power, axis=0)
    pstd = np.std(power, axis=0)
    expected = pkfunc(np.asarray(result.bin_centres))
    mask = np.isfinite(pstd[1:]) & (pstd[1:] > 0)
    zscore = np.abs(pmean[1:][mask] - expected[1:][mask]) / (
        pstd[1:][mask] / np.sqrt(nrealizations)
    )
    _assert_reasonable_power_recovery(zscore)


def test_jax_powerbox_delta_k_rejects_negative_power() -> None:
    pb = jpb.PowerBox(
        shape=(8, 10),
        pk=lambda k: -jnp.ones_like(k),
        size=(4.0, 5.0),
        key=jax.random.key(16),
    )
    with pytest.raises(ValueError, match="negative values"):
        pb.delta_k()
