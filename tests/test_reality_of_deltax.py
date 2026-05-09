"""Tests that deltax is real under different assumptions."""

import numpy as np
import pytest

from powerbox import PowerBox
from powerbox.dft import ifft


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("ncells", [16, 17])
@pytest.mark.parametrize("ab", [(0, 1), (0, 2 * np.pi)])
def test_deltax_is_real(ndim, ncells, ab):
    pb = PowerBox(pk=lambda k: 1, boxlength=1, N=ncells, dim=ndim, seed=1234)

    dk = pb.delta_k()

    deltax = ifft(
        dk,
        L=1,
        a=ab[0],
        b=ab[1],
    )[0]

    np.testing.assert_allclose(deltax.imag, 0, atol=1e-12)
