import numpy as np
import os
import inspect
import sys

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
sys.path.insert(0, LOCATION)

from powerbox.tools import angular_average
from powerbox.dft import fft, ifft

# SETUP
N = 1000
L = 10.
dx = L/N
x = np.arange(-L/2, L/2, dx)[:N]
X, Y = np.meshgrid(x, x)
a_squared = 1
fx = np.exp(-np.pi*a_squared*(X ** 2 + Y ** 2))


def run_roundtrip(a, b):
    Fx, freq = fft(fx, L=L, a=a, b=b)

    Lk = -2*np.min(freq)
    fx_, x_ = ifft(Fx, Lk=Lk, a=a, b=b)
    assert np.max(np.real((fx_ - fx))) < 1e-10  # Test FT result
    assert np.max(x_[0] - x) < 1e-10  # Test x-grid


def test_roundtrip_unitary_ordinary():
    run_roundtrip(0, 2*np.pi)


def test_roundtrip_unitary_angular():
    run_roundtrip(0, 1)


def test_roundtrip_non_unitary_angular():
    run_roundtrip(1, 1)


def test_forward_unitary_ordinary():
    a, b = 0, 2*np.pi

    Fx, freq, grid = fft(fx, L=L, a=a, b=b, ret_cubegrid=True)
    Fx_anl_fc = lambda k: (1./a_squared)*np.exp(-np.pi*k ** 2/a_squared)

    # Analytic transform
    Fx_anl = Fx_anl_fc(grid)

    Fx_circ = angular_average(np.abs(Fx), grid, int(N/2.2))[0]
    Fx_anl_circ = angular_average(Fx_anl, grid, int(N/2.2))[0]

    assert np.max(np.abs(Fx_circ - Fx_anl_circ)) < 1e-10


def test_forward_unitary_angular():
    a, b = 0, 1.

    Fx, freq, grid = fft(fx, L=L, a=a, b=b, ret_cubegrid=True)
    Fx_anl_fc = lambda k: 1./(2*np.pi*a_squared)*np.exp(-(1./(4*np.pi))*k ** 2/a_squared)

    # Analytic transform
    Fx_anl = Fx_anl_fc(grid)

    Fx_circ = angular_average(np.abs(Fx), grid, int(N/2.2))[0]
    Fx_anl_circ = angular_average(Fx_anl, grid, int(N/2.2))[0]

    assert np.max(np.abs(Fx_circ - Fx_anl_circ)) < 1e-10


def test_forward_non_unitary_angular():
    a, b = 1, 1.

    Fx, freq, grid = fft(fx, L=L, a=a, b=b, ret_cubegrid=True)
    Fx_anl_fc = lambda k: (1./a_squared)*np.exp(-k ** 2/(4*np.pi*a_squared))

    # Analytic transform
    Fx_anl = Fx_anl_fc(grid)

    Fx_circ = angular_average(np.abs(Fx), grid, int(N/2.2))[0]
    Fx_anl_circ = angular_average(Fx_anl, grid, int(N/2.2))[0]

    assert np.max(np.abs(Fx_circ - Fx_anl_circ)) < 1e-10


def test_mixed_unitary_ordinary_unitary_angular_1d_fb():
    N = 1000
    L = 10.
    dx = L/N
    x = np.arange(-L/2, L/2, dx)[:N]

    alpha = np.pi
    fx = np.exp(-alpha*x ** 2)

    Fk, freq = fft(fx, L=L, a=0, b=2*np.pi, )
    Lk = -2*np.min(freq)
    fx_, x_ = ifft(Fk, Lk=Lk, a=0, b=1)
    assert np.max(np.abs(fx_ - np.exp(-np.pi*(x_[0]/(2*np.pi)) ** 2)/np.sqrt(2*np.pi))) < 1e-10


def test_mixed_unitary_ordinary_unitary_angular_1d_bf():
    N = 1000
    Lk = 10.
    dk = Lk/N
    k = np.arange(-Lk/2, Lk/2, dk)[:N]

    alpha = np.pi
    Fk = np.exp(-alpha*k ** 2)

    fx, x = ifft(Fk, Lk=Lk, a=0, b=1)
    Fk_, k_ = fft(fx, -2*np.min(x), a=0, b=2*np.pi)
    assert np.max(np.abs(Fk_ - np.sqrt(2*np.pi)*np.exp(-alpha*((2*np.pi)*k_[0]) ** 2))) < 1e-10


def test_mixed_unitary_ordinary_unitary_angular_2d_fb():
    N = 1000
    L = 10.
    dx = L/N
    x = np.arange(-L/2, L/2, dx)[:N]
    X, Y = np.meshgrid(x, x)

    a_squared = 1
    fx = np.exp(-np.pi*a_squared*(X ** 2 + Y ** 2))

    Fk, freq = fft(fx, L=L, a=0, b=2*np.pi)
    Lk = -2*np.min(freq)
    fx_, x_, xgrid = ifft(Fk, Lk=Lk, a=0, b=1, ret_cubegrid=True)

    fx_, bins = angular_average(fx_, xgrid, 200)

    print(np.max(np.abs(fx_ - np.exp(-np.pi*(bins/(2*np.pi)) ** 2)/(2*np.pi))))
    assert np.max(np.abs(fx_ - np.exp(-np.pi*(bins/(2*np.pi)) ** 2)/(2*np.pi))) < 1e-4


def test_mixed_unitary_ordinary_unitary_angular_2d_bf():
    N = 1000
    Lk = 10.
    dk = Lk/N
    k = np.arange(-Lk/2, Lk/2, dk)[:N]
    KX, KY = np.meshgrid(k, k)

    alpha = np.pi
    Fk = np.exp(-alpha*(KX ** 2 + KY ** 2))

    fx, x = ifft(Fk, Lk=Lk, a=0, b=1)
    Fk_, k_, kgrid = fft(fx, -2*np.min(x), a=0, b=2*np.pi, ret_cubegrid=True)

    Fk_, bins = angular_average(Fk_, kgrid, 200)
    assert np.max(np.abs(Fk_ - 2*np.pi*np.exp(-alpha*((2*np.pi)*bins) ** 2)))
