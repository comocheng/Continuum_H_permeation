"""The resistance fractions that define the SA regimes are degrees of rate control.

A regime label answers "which step limits the flux". The quantity that answers it is
the degree of rate control X_i = d ln J / d ln k_i (Campbell): how much the flux
rises when step i alone is made faster. These tests check every fraction the regime
labels are built from against a finite-difference X_i, so a fraction computed in
inconsistent units (for example a surface resistance not on the sqrt(P) scale of the
layers, or an L5 split taken from the linear pressure drop) fails here rather than
silently relabelling SA clusters.
"""
import numpy as np
import pytest

from calculations.surface_kinetics import (
    solve_steady_state_flux_L1L6, solve_steady_state_flux_direct,
    calculate_path_flux_L346_v2,
)
from calculations.config.model_config import DEFAULT_PARAMS_LEVEL5
from calculations.sensitivity import (
    level5_model_wrapper, series_rate_control_L5,
)

EPS = 1e-5


def drc(f, x):
    """d ln f / d ln x by central difference."""
    return (np.log(f(x * (1 + EPS))) - np.log(f(x * (1 - EPS)))) / (
        np.log(1 + EPS) - np.log(1 - EPS))


# (P_up, k_diss, K_eq) spanning surface-, mixed- and transport-limited walls
SURFACE_CASES = [
    (1e5, 1e-7, 1e-4),
    (1e5, 1e-10, 1e-4),
    (1e-2, 1e-7, 1e-4),
    (1e3, 1e-5, 1e-2),
    (1e5, 3e-12, 1e-6),
]
D_M, K_S, L_M = 1e-10, 0.05, 1e-3
D_OX, K_OX, L_OX = 1e-17, 0.3, 1e-6


@pytest.mark.parametrize("P_up,k,K", SURFACE_CASES)
def test_L1L6_fractions_are_degrees_of_rate_control(P_up, k, K):
    r = solve_steady_state_flux_L1L6(P_up, 0.0, L_M, k, K, D_M, K_S)
    fr = r['resistances']
    X_s = drc(lambda kk: solve_steady_state_flux_L1L6(
        P_up, 0.0, L_M, kk, K, D_M, K_S)['J_ss'], k)
    X_m = drc(lambda d: solve_steady_state_flux_L1L6(
        P_up, 0.0, L_M, k, K, d, K_S)['J_ss'], D_M)
    assert fr['fraction_surface'] == pytest.approx(X_s, abs=1e-4)
    assert fr['fraction_metal'] == pytest.approx(X_m, abs=1e-4)
    assert X_s + X_m == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("P_up,k,K", SURFACE_CASES)
def test_L2L6_fractions_are_degrees_of_rate_control(P_up, k, K):
    def J(k_=k, dox=D_OX, dm=D_M):
        return solve_steady_state_flux_direct(
            P_up, 0.0, L_M, k_, K, dox, K_OX, L_OX, dm, K_S)['J_ss']
    fr = solve_steady_state_flux_direct(
        P_up, 0.0, L_M, k, K, D_OX, K_OX, L_OX, D_M, K_S)['resistances']
    assert fr['fraction_surface'] == pytest.approx(drc(lambda x: J(k_=x), k), abs=1e-4)
    assert fr['fraction_oxide'] == pytest.approx(drc(lambda x: J(dox=x), D_OX), abs=1e-4)
    assert fr['fraction_metal'] == pytest.approx(drc(lambda x: J(dm=x), D_M), abs=1e-4)


def test_pinhole_path_surface_fraction_is_rate_control():
    micro = {'grain_size': 1e-4, 'gb_type': 'LAGB', 'gb_thickness': 5e-10}

    def J(k):
        return calculate_path_flux_L346_v2(
            1e5, 0.0, L_M, 873.0, 1e-7, 1e-4, np.inf, D_M, K_S, micro,
            path_type='pinhole', enable_gb_enhancement=False,
            k_diss_metal=k, K_eq_metal=1e-3)['flux']
    r = calculate_path_flux_L346_v2(
        1e5, 0.0, L_M, 873.0, 1e-7, 1e-4, np.inf, D_M, K_S, micro,
        path_type='pinhole', enable_gb_enhancement=False,
        k_diss_metal=1e-9, K_eq_metal=1e-3)
    assert r['resistances']['fraction_surface'] == pytest.approx(drc(J, 1e-9), abs=1e-4)


def test_surface_that_costs_a_factor_of_nine_is_not_reported_as_minor():
    """The docs' worked example: a surface that cuts the 316L flux ninefold.

    If the surface costs a factor of 9, making it faster is what raises the flux,
    so it must carry most of the rate control.
    """
    P_up, k, K = 1e5, 3e-12, 1e-4
    r = solve_steady_state_flux_L1L6(P_up, 0.0, L_M, k, K, D_M, K_S)
    J_sieverts = D_M * K_S * np.sqrt(P_up) / L_M
    assert J_sieverts / r['J_ss'] > 5
    assert r['rate_limiting'] == 'surface'


@pytest.mark.parametrize("ratio", [0.01, 0.2, 1 / 3, 0.45, 0.9])
def test_L5_series_split_closed_form(ratio):
    P = 1e5
    assert series_rate_control_L5(P, ratio * P) == pytest.approx(
        (1 - ratio) / (1 + ratio))


def test_L5_layers_hand_over_at_a_third_not_a_half():
    # The linear pressure-drop share would say 'oxide' anywhere P_int < P_up/2.
    assert series_rate_control_L5(1e5, 0.4e5) < 0.5
    assert series_rate_control_L5(1e5, 1e5 / 3) == pytest.approx(0.5)


@pytest.mark.parametrize("params", [
    {},                                                   # default 316L wall
    {'oxide_thickness': 3e-5},                            # thick oxide
    {'oxide_thickness': 1e-5, 'D_ox_ref': 3e-17},         # mixed
])
def test_L5_wrapper_series_fractions_are_rate_control(params):
    sealed = {'f_pinhole': 1e-14, 'f_crack': 1e-14, 'f_gb_defect': 1e-14,
              'enable_gb_enhancement': False, **params}
    rec = level5_model_wrapper(sealed)
    base = {**DEFAULT_PARAMS_LEVEL5, **sealed}
    X_ox = drc(lambda d: level5_model_wrapper({**sealed, 'D_ox_ref': d})['flux'],
               base['D_ox_ref'])
    X_m = drc(lambda d: level5_model_wrapper({**sealed, 'D_ref': d})['flux'],
              base['D_ref'])
    assert X_ox + X_m == pytest.approx(1.0, abs=1e-3)
    assert rec['frac_oxide'] == pytest.approx(X_ox, abs=1e-3)
    assert rec['frac_metal'] == pytest.approx(X_m, abs=1e-3)
