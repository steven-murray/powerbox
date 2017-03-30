import numpy as np
import os
import inspect
import sys

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
sys.path.insert(0, LOCATION)

from powerbox import LogNormalPowerBox,PowerBox, get_power

def test_ln_vs_straight():
    # Set up two boxes with exactly the same parameters
    pb = PowerBox(128,lambda u : 100.*u**-2., dim=3,seed=1234,boxlength=100.)
    ln_pb = LogNormalPowerBox(128,lambda u : 100.*u**-2., dim=3,seed=1234,boxlength=100.)


    pk = get_power(pb.delta_x(),pb.boxlength)[0]
    ln_pk = get_power(ln_pb.delta_x(), pb.boxlength)[0]

    pk = pk[1:-1]
    ln_pk = ln_pk[1:-1]
    print np.mean(np.abs((pk-ln_pk)/pk)), np.abs((pk-ln_pk)/pk)
    assert np.mean(np.abs((pk-ln_pk)/pk)) < 2e-1   # 10% agreement


def test_ln_vs_straight_standard_freq():
    # Set up two boxes with exactly the same parameters
    pb = PowerBox(128,lambda u : 12.*u**-2., dim=3,seed=1234,boxlength=1200.,a=0,b=2*np.pi)
    ln_pb = LogNormalPowerBox(128,lambda u : 12.*u**-2., dim=3,seed=1234,boxlength=1200.,a=0,b=2*np.pi)


    pk = get_power(pb.delta_x(),pb.boxlength,a=0,b=2*np.pi)[0]
    ln_pk = get_power(ln_pb.delta_x(), pb.boxlength,a=0,b=2*np.pi)[0]

    pk = pk[1:-1]
    ln_pk = ln_pk[1:-1]
    print np.mean(np.abs((pk-ln_pk)/pk)), np.abs((pk-ln_pk)/pk)
    assert np.mean(np.abs((pk-ln_pk)/pk)) < 2e-1   # 10% agreement
