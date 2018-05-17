import numpy as np
from powerbox.tools import angular_average_nd, angular_average


def test_angular_avg_nd_3():
    x = np.linspace(-3,3,400)
    X,Y = np.meshgrid(x,x)
    r2 = X**2 + Y**2
    P = r2**-1.
    P = np.repeat(P,100).reshape(400,400,100)
    freq = [x,x,np.linspace(-2,2,100)]
    p_k, k_av_bins = angular_average_nd(P,freq,bins=50,n=2)
    print(p_k[6:,0], k_av_bins[6:]**-2.)
    assert np.max(np.abs((p_k[6:,0] - k_av_bins[6:]**-2.)/k_av_bins[6:]**-2.)) <  0.05


def test_angular_avg_nd_4_2():
    x = np.linspace(-3,3,200)
    X,Y = np.meshgrid(x,x)
    r2 = X**2 + Y**2
    P = r2**-1.
    P = np.repeat(P, 10).reshape(200,200,10)
    P = np.repeat(P, 10).reshape(200, 200, 10, 10)

    freq = [x,x,np.linspace(-2,2,10), np.linspace(-2,2,10)]
    p_k, k_av_bins = angular_average_nd(P,freq,bins=50,n=2)

    print(np.abs((p_k[7:,0,0] - k_av_bins[7:]**-2.)/k_av_bins[7:]**-2.))
    assert np.max(np.abs((p_k[6:,0,0] - k_av_bins[6:]**-2.)/k_av_bins[6:]**-2.)) <  0.06


def test_null_variance_2d():
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    P = np.ones_like(r2)
    ave, coord, var = angular_average(P, np.sqrt(r2), bins=np.linspace(0,x.max(), 20), get_variance=True)
    # Ensure that the variance is very small. It won't be exactly zero
    assert np.all(var==0)


def test_variance_2d():
    x = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    P = np.ones_like(r2)
    P += np.random.normal(scale=1, size=(len(x), len(x)))
    ave, coord, var = angular_average(P, np.sqrt(r2), bins=np.linspace(0,x.max(), 20), get_variance=True)
    print(np.diff(var))
    assert np.all(np.diff(var)<=0)

