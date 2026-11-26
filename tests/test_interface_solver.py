"""The Level 2b oxide/metal interface solve at the edges the SA now samples.

Log-uniform sampling puts half the P_upstream draws below 1 Pa and reaches 1 nm
oxides at 1e7 Pa. Both broke the brentq solve: an absolute xtol larger than the root
at low pressure, and a fallback that assumed every unbracketable root was
oxide-dominated. Each case is checked against the exact root of the quadratic in
sqrt(P_int).
"""
import numpy as np
import pytest

from calculations.interface_solver import calculate_oxide_metal_system


def exact(P_up, P_down, ox, m):
    a = ox['D_ox'] * ox['K_ox'] / ox['thickness']
    b = m['D_metal'] * m['K_s_metal'] / m['thickness']
    c = a * P_up + b * np.sqrt(P_down)
    s = 2 * c / (b + np.sqrt(b * b + 4 * a * c))
    return s * s, b * (s - np.sqrt(P_down))


CASES = {
    # P_int ~ 1e-14 Pa: below the old absolute xtol of 1e-12
    'low pressure': (1e-6, 0.0, {'D_ox': 1e-16, 'K_ox': 1.0, 'thickness': 1e-6},
                     {'D_metal': 1e-9, 'K_s_metal': 0.05, 'thickness': 1e-3}),
    # oxide so permeable that P_up - P_int < 1e-10 P_up: root above the bracket
    'metal-dominated': (1e7, 0.0, {'D_ox': 1e-15, 'K_ox': 300.0, 'thickness': 1e-9},
                        {'D_metal': 1e-12, 'K_s_metal': 0.05, 'thickness': 5e-3}),
    # oxide so tight that P_int < 1e-10 P_up: root below the bracket
    'oxide-dominated': (1e-5, 0.0, {'D_ox': 1e-22, 'K_ox': 1e-3, 'thickness': 1e-5},
                        {'D_metal': 1e-8, 'K_s_metal': 0.05, 'thickness': 1e-4}),
    'ordinary': (1e5, 10.0, {'D_ox': 1e-17, 'K_ox': 287.0, 'thickness': 1e-6},
                 {'D_metal': 2e-10, 'K_s_metal': 0.058, 'thickness': 1e-3}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_interface_matches_exact_root(case):
    P_up, P_down, ox, m = CASES[case]
    r = calculate_oxide_metal_system(P_up, P_down, ox, m)
    P_int, J = exact(P_up, P_down, ox, m)
    assert r['P_interface'] == pytest.approx(P_int, rel=1e-6)
    assert r['flux'] == pytest.approx(J, rel=1e-6)


def test_random_walls_across_sa_ranges():
    rng = np.random.default_rng(0)
    for _ in range(500):
        ox = {'D_ox': 10 ** rng.uniform(-20, -14), 'K_ox': 10 ** rng.uniform(-2, 3),
              'thickness': 10 ** rng.uniform(-9, -5)}
        m = {'D_metal': 10 ** rng.uniform(-12, -7),
             'K_s_metal': 10 ** rng.uniform(-3, 0),
             'thickness': 10 ** rng.uniform(-4, -2.3)}
        P = 10 ** rng.uniform(-7, 7)
        r = calculate_oxide_metal_system(P, 0.0, ox, m)
        _, J = exact(P, 0.0, ox, m)
        assert r['flux'] == pytest.approx(J, rel=1e-3)
