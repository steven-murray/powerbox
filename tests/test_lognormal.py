import numpy as np

from powerbox import LogNormalPowerBox, PowerBox, get_power


def test_ln_vs_straight():
    # Set up two boxes with exactly the same parameters
    pb = PowerBox(128, lambda u: 100.0 * u**-2.0, dim=3, seed=1234, boxlength=100.0)
    ln_pb = LogNormalPowerBox(
        128, lambda u: 100.0 * u**-2.0, dim=3, seed=1234, boxlength=100.0
    )

    pk = get_power(pb.delta_x(), pb.boxlength)[0]
    ln_pk = get_power(ln_pb.delta_x(), pb.boxlength)[0]

    pk = pk[1:-1]
    ln_pk = ln_pk[1:-1]
    print(np.mean(np.abs((pk - ln_pk) / pk)), np.abs((pk - ln_pk) / pk))
    assert np.mean(np.abs((pk - ln_pk) / pk)) < 2e-1  # 10% agreement


def test_ln_vs_straight_standard_freq():
    # Set up two boxes with exactly the same parameters
    pb = PowerBox(
        128,
        lambda u: 12.0 * u**-2.0,
        dim=3,
        seed=1234,
        boxlength=1200.0,
        a=0,
        b=2 * np.pi,
    )
    ln_pb = LogNormalPowerBox(
        128,
        lambda u: 12.0 * u**-2.0,
        dim=3,
        seed=1234,
        boxlength=1200.0,
        a=0,
        b=2 * np.pi,
    )

    pk = get_power(pb.delta_x(), pb.boxlength, a=0, b=2 * np.pi)[0]
    ln_pk = get_power(ln_pb.delta_x(), pb.boxlength, a=0, b=2 * np.pi)[0]

    pk = pk[1:-1]
    ln_pk = ln_pk[1:-1]
    print(np.mean(np.abs((pk - ln_pk) / pk)), np.abs((pk - ln_pk) / pk))
    assert np.mean(np.abs((pk - ln_pk) / pk)) < 2e-1  # 10% agreement
