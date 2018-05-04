import os
import inspect
import sys

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
sys.path.insert(0, LOCATION)

import numpy as np
from powerbox.tools import angular_average_nd

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
