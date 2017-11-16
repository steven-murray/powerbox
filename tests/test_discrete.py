import numpy as np
import os
import inspect
import sys

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
sys.path.insert(0, LOCATION)

from powerbox import PowerBox, LogNormalPowerBox, get_power


def test_discrete_power_gaussian():
    pb = PowerBox(N=512,dim=2,boxlength=100., pk = lambda u : 0.1*u**-1.5, ensure_physical=True)

    sample = pb.create_discrete_sample(nbar=1000.)
    power, bins = get_power(sample,pb.boxlength,N=pb.N)


    res =  np.mean(np.abs(power[50:-50] / (0.1*bins[50:-50]**-1.5) -1 ) )

    print(res)
    assert res < 1e-1

def test_discrete_power_lognormal():
    pb = LogNormalPowerBox(N=512, dim=2, boxlength=100., pk=lambda u: 0.1*u ** -1.5, ensure_physical=True)

    sample = pb.create_discrete_sample(nbar=1000.)
    power, bins = get_power(sample, pb.boxlength, N=pb.N)

    res = np.mean(np.abs(power[50:-50]/(0.1*bins[50:-50] ** -1.5) - 1))

    print(res)
    assert res < 1e-1