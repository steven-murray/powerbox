import numpy as np
from scipy.ndimage import gaussian_filter
import os
import inspect
import sys

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
sys.path.insert(0, LOCATION)

from powerbox import PowerBox


def test_resolution():
    var = [0]*6
    for i in range(6):
        pb = PowerBox(64*2**i, dim=2, pk=lambda k: 1.0*k ** -2., boxlength=1.0, angular_freq=True)
        var[i] = np.var(gaussian_filter(pb.delta_x,sigma=2**i,mode='wrap'))
    print var/var[0]
    assert np.allclose(var/var[0],1,atol=1e-2)
