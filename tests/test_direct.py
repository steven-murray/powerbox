import numpy as np

from powerbox import PowerBox

N = 5


def ensure_hermitian(b):
    revidx = (slice(None, None, -1),) * len(b.shape)

    if len(b) % 2 == 0:
        cutidx = (slice(1, None, None),) * len(b.shape)
        b = b[cutidx]

    print(b - b[revidx])
    assert np.allclose(np.real(b - b[revidx]), 0)
    assert np.allclose(np.imag(b + b[revidx]), 0)


def ensure_reality_elementwise(x):
    if not np.allclose(np.abs(np.imag(x) / np.real(x)), 0, atol=0.01, rtol=0.01):
        print(
            "Maximum contribution of imaginary part in any element: ",
            np.max(np.abs(np.imag(x) / np.real(x))),
        )
        return False
    else:
        return True


def ensure_reality(x):
    val = np.sum(np.abs(x)) / np.sum(np.abs(np.real(x)))
    if np.isclose(val, 1, rtol=5e-3):
        return True
    else:
        print("Total fractional contribution of imaginary parts", val - 1)
        return False


class TestDirect:
    def setup_method(self, test_method):
        self.pb = PowerBox(N, lambda k: k**-2.0, dim=1)

    def test_hermitian(self):
        ensure_hermitian(self.pb.delta_k())

    def test_reality_elementwise(self):
        ensure_reality_elementwise(self.pb.delta_x())

    def test_reality(self):
        ensure_reality(self.pb.delta_x())


class TestDirect2(TestDirect):
    def setup_method(self, test_method):
        self.pb = PowerBox(N, lambda k: k**-2.0, dim=2)


class TestDirect3(TestDirect):
    def setup_method(self, test_method):
        self.pb = PowerBox(N, lambda k: k**-2.0, dim=3)


class TestDirect4(TestDirect):
    def setup_method(self, test_method):
        self.pb = PowerBox(N, lambda k: k**-2.0, dim=4)


class TestDirectEven(TestDirect):
    def setup_method(self, test_method):
        self.pb = PowerBox(N - 1, lambda k: k**-2.0, dim=2)

    def test_reality_elementwise(self):
        pass  # It won't be element-wise correct for even case.

    def test_reality(self):
        ensure_reality(self.pb.delta_x())
