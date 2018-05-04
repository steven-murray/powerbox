import numpy as np
import os
import inspect
import sys

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
sys.path.insert(0, LOCATION)

from powerbox import PowerBox, get_power

def test_power1d():
    p = [0]*40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0*k ** -2., boxlength=1.0, a=0,b=1 )
        p[i], k = get_power(pb.delta_x(), pb.boxlength,a=0,b=1)

    print(np.mean(np.array(p),axis=0)/(1.0*k**-2.))
    assert np.allclose(np.mean(np.array(p),axis=0)[2000:],1.0*k[2000:]**-2.,rtol=2)

def test_power1d_n3():
    p = [0]*40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0*k ** -3., boxlength=1.0, b=1 )
        p[i], k = get_power(pb.delta_x(), pb.boxlength,b=1)

    print(np.mean(np.array(p), axis=0)/(1.0*k ** -2.))
    assert np.allclose(np.mean(np.array(p), axis=0)[2000:], 1.0*k[2000:] ** -2., rtol=2)


def test_power1d_bigL():
    p = [0]*40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0*k ** -3., boxlength=10.0, b=1 )
        p[i], k = get_power(pb.delta_x(), pb.boxlength,b=1)

    print(np.mean(np.array(p), axis=0)/(1.0*k ** -2.))
    assert np.allclose(np.mean(np.array(p), axis=0)[2000:], 1.0*k[2000:] ** -2., rtol=2)


def test_power1d_ordinary_freq():
    p = [0]*40
    for i in range(40):
        pb = PowerBox(8001, dim=1, pk=lambda k: 1.0*k ** -3., boxlength=1.0 )
        p[i], k = get_power(pb.delta_x(), pb.boxlength)

    print(np.mean(np.array(p), axis=0)/(1.0*k ** -2.))
    assert np.allclose(np.mean(np.array(p), axis=0)[2000:], 1.0*k[2000:] ** -2., rtol=2)



def test_power1d_halfN():
    p = [0]*40
    for i in range(40):
        pb = PowerBox(4001, dim=1, pk=lambda k: 1.0*k ** -3., boxlength=1.0, b=1 )
        p[i], k = get_power(pb.delta_x(), pb.boxlength,b=1)

    print(np.mean(np.array(p), axis=0)/(1.0*k ** -2.))
    assert np.allclose(np.mean(np.array(p), axis=0)[1000:], 1.0*k[1000:] ** -2., rtol=2)


def test_power2d():
    p = [0]*5
    for i in range(5):
        pb = PowerBox(200, dim=2, pk=lambda k: 1.0*k ** -2., boxlength=1.0, b=1)
        p[i], k = get_power(pb.delta_x(), pb.boxlength,b=1)

    print(np.mean(np.array(p),axis=0)/(1.0*k**-2.))
    assert np.allclose(np.mean(np.array(p),axis=0)[100:],1.0*k[100:]**-2.,rtol=2)


def test_power3d():
    pb = PowerBox(50, dim=2, pk=lambda k: 1.0*k ** -2., boxlength=1.0, b=1)
    p, k = get_power(pb.delta_x(), pb.boxlength,b=1)

    print(p/(1.0*k**-2.))
    assert np.allclose(p,1.0*k**-2.,rtol=2)
