import numpy as np
import os
import inspect
import sys
import pytest

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
sys.path.insert(0, LOCATION)

from powerbox.powerbox import PowerBox, _make_hermitian
from powerbox import dft
from itertools import product

@pytest.fixture(scope='function', params=product([10,11], [1, 2, 3, 4]))
def ghbox(request):
    n, ndim = request.param
    print(f"N={n}, ndim={ndim}")
    pb = PowerBox(n, lambda k: k**-2, dim=ndim)
    return pb.gauss_hermitian()



def ensure_hermitian(b):

    revidx = (slice(None, None, -1),)*len(b.shape)

    if len(b)%2==0:
        cutidx = (slice(1,None,None),)*len(b.shape)
        b = b[cutidx]

    a = b-np.conj(b[revidx])
    np.testing.assert_allclose(a.real, 0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(a.imag, 0, rtol=0, atol=1e-13)


def ensure_reality_elementwise(x):
    np.testing.assert_allclose(x.imag, 0, rtol=0, atol=1e-13)


# def ensure_reality(x):
#     val = np.sum(np.abs(x))/np.sum(np.abs(np.real(x)))
#     if np.isclose(val, 1,rtol=5e-3):
#         return True
#     else:
#         print("Total fractional contribution of imaginary parts", val - 1)
#         return False

class TestHermitian:

    def test_hermitian(self, ghbox):
        ensure_hermitian(ghbox)

    def test_reality_of_ifft(self, ghbox):
        #xx = dft.ifft(ghbox)[0]
        n = len(ghbox)
        if not n%2:
            n += 1

        xx = np.fft.ifftn(np.fft.ifftshift(ghbox), s=[n]*ghbox.ndim)
        ensure_reality_elementwise(xx)

# class TestDirect(object):
#     def setup_method(self, test_method):
#         self.pb = PowerBox(N,lambda k: k**-2.,dim=1)

#     def test_hermitian(self):
#         ensure_hermitian(self.pb.delta_k())

#     def test_reality_elementwise(self):
#         assert ensure_reality_elementwise(self.pb.delta_x())

#     # def test_reality(self):
#     #     ensure_reality(self.pb.delta_x())


# class TestDirect2(TestDirect):
#     def setup_method(self, test_method):
#         self.pb = PowerBox(N,lambda k: k**-2.,dim=2)


# class TestDirect3(TestDirect):
#     def setup_method(self, test_method):
#         self.pb = PowerBox(N, lambda k: k ** -2., dim=3)


# class TestDirect4(TestDirect):
#     def setup_method(self, test_method):
#         self.pb = PowerBox(N, lambda k: k ** -2., dim=4)


# class TestDirectEven(TestDirect):
#     def setup_method(self, test_method):
#         self.pb = PowerBox(N-1, lambda k: k ** -2., dim=2)

#     def test_reality_elementwise(self):
#         return True ## It won't be element-wise correct for even case.

#     def test_reality(self):
#         ensure_reality(self.pb.delta_x())



