import numpy as np
import os
import inspect
import sys

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
sys.path.insert(0, LOCATION)

from powerbox import PowerBox, LogNormalPowerBox, get_power

def test_discrete_power_gaussian():
    # Supply `seed` to ensure that discrete sample corresponds to box.
    pb = PowerBox(N=512,dim=2,boxlength=100., pk = lambda u : 0.1*u**-1.5,
        ensure_physical=True, seed=2468)

    sample = pb.create_discrete_sample(nbar=1000.)
    power, bins = get_power(sample,pb.boxlength,N=pb.N)

    res =  np.mean(np.abs(power[50:-50] / (0.1*bins[50:-50]**-1.5) -1 ) )

    assert res < 1e-1

    # This re-grids the discrete sample into a box, basically to verify the
    # indexing used by meshgrid within `create_discrete_sample`.
    N = [pb.N] * pb.dim
    L = [pb.boxlength] * pb.dim
    edges = [np.linspace(-_L/2., _L/2., _n + 1) for _L, _n in zip(L, N)]
    delta_samp = np.histogramdd(sample, bins=edges, weights=None)[0].astype("float")

    # Check cross spectrum and assert a strong correlation
    cross, bins = get_power(delta_samp,pb.boxlength,deltax2=pb.delta_x())
    p2, bins = get_power(pb.delta_x(), pb.boxlength)
    corr = cross / np.sqrt(power) / np.sqrt(p2)
    corr_bar = np.mean(corr[np.isfinite(corr)])
    assert corr_bar > 10

def test_discrete_power_lognormal():
    pb = LogNormalPowerBox(N=512, dim=2, boxlength=100., pk=lambda u: 0.1*u ** -1.5, ensure_physical=True)

    sample = pb.create_discrete_sample(nbar=1000.)
    power, bins = get_power(sample, pb.boxlength, N=pb.N)

    res = np.mean(np.abs(power[50:-50]/(0.1*bins[50:-50] ** -1.5) - 1))

    assert res < 1e-1

if __name__ == '__main__':
    test_discrete_power_gaussian()
    test_discrete_power_lognormal()
