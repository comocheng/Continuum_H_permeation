"""The regime labels in calculations.wall_labels.

A label makes two claims about a point: which path carries most of the wall's flux, and which
process controls the wall's rate. These tests pin the bookkeeping the module adds on top of the
solvers: that a path dominates only above half the flux, that each surface is named for the
surface its path actually has, that the Model 1 paths re-solved one by one add up to the flux the
wall solver returned, that a dropped defect path is never read as a defect-free wall, and that the
whole-wall degrees of rate control are what a finite difference of the wall's flux gives.
"""
import math

import numpy as np
import pytest

from calculations.config.model_config import (CONDITIONS, METALS, MICROSTRUCTURE, MODEL_OPTIONS, OXIDE_DEFECTS,
                                              build_simulation_config, get_surface_kinetics_from_config)
from calculations.oxide_permeation import get_oxide_properties_at_T, get_metal_properties_at_T
from calculations.parallel_oxide_defect_paths import (calculate_parallel_path_flux,
                                                      calculate_parallel_path_flux_defective_metal)
from calculations.surface_kinetics import (get_all_properties, calculate_mixed_defect_flux_L6,
                                           calculate_full_model_flux_L346_v2)
from calculations.wall_labels import (CONTROL, DOMINANCE, MIXED_CONTROL, NO_DOMINANT_PATH, SURFACE_OF,
                                      X_KEYS, compose, controlling, defect_fractions, describe, fitted,
                                      label_from_paths, label_model2, label_model1, model1_paths,
                                      model2_paths, parse, rate_control, wall_from_paths, wall_model1,
                                      wall_model2)

SIM = build_simulation_config()
MICRO_M1 = dict(microstructure_params=MICROSTRUCTURE, enable_gb_enhancement=MODEL_OPTIONS['enable_gb_enhancement'])

# The operating point, and the low-pressure end where the oxide (Model 1) or the surface
# (Model 2) holds hydrogen back, so both kinds of label are exercised.
POINTS = [(CONDITIONS['T_operating'], CONDITIONS['P_upstream']),
          (CONDITIONS['T_operating'], CONDITIONS['P_range'][0])]


def layers(T):
    return (dict(get_oxide_properties_at_T(SIM['oxide_name'], T), thickness=SIM['L_oxide']),
            dict(get_metal_properties_at_T(SIM['metal_name'], T), thickness=SIM['L_metal']))


def model2_wall(level, T, P, defects):
    p  = get_all_properties(SIM['oxide_name'], SIM['metal_name'], T)
    sm = get_surface_kinetics_from_config(SIM['metal_name'], T, METALS)
    args = (P, SIM['P_downstream'], SIM['L_metal'])
    kin  = dict(k_diss_metal=sm['k_diss'], K_eq_metal=sm['K_eq'])
    if level == 'L3L6':
        return calculate_mixed_defect_flux_L6(*args, p['k_diss'], p['K_eq'], p['D_ox'], p['K_ox'], SIM['L_oxide'],
                                              p['D_m'], p['K_s_m'], defects, **kin)
    return calculate_full_model_flux_L346_v2(*args, T, p['k_diss'], p['K_eq'], p['D_ox'], p['K_ox'], SIM['L_oxide'],
                                             p['D_m'], p['K_s_m'], SIM['microstructure'], defects, **kin)


@pytest.mark.parametrize('step,path', [('oxide surface', None), ('metal diffusion', 'pinhole'),
                                       (MIXED_CONTROL, 'intact'), ('oxide diffusion', 'grain_boundary'),
                                       ('metal surface', NO_DOMINANT_PATH)])
def test_compose_and_parse_are_inverses(step, path):
    from calculations.wall_labels import PATH_NAMES
    assert parse(compose(step, path)) == (PATH_NAMES.get(path, path) if path else None, step)


def two_paths(share, X_intact, X_pinhole):
    """A wall of flux 1 whose pinholes carry ``share`` of it."""
    return {'intact':  {'contribution': 1 - share, **rate_control(**X_intact)},
            'pinhole': {'contribution': share,     **rate_control(**X_pinhole)}}


def test_a_path_dominates_only_above_half_the_flux():
    """At exactly half neither path dominates; a hair above, the larger one does."""
    oxide, metal = {'oxide_diffusion': 1.0}, {'metal_surface': 1.0}
    assert label_from_paths(two_paths(DOMINANCE, oxide, metal), 1.0) == f'{NO_DOMINANT_PATH}/{MIXED_CONTROL}'
    assert label_from_paths(two_paths(DOMINANCE + 1e-3, oxide, metal), 1.0) == 'pinholes/metal surface'
    assert label_from_paths(two_paths(DOMINANCE - 1e-3, oxide, metal), 1.0) == 'intact/oxide diffusion'
    assert label_from_paths(two_paths(0.3, oxide, metal), math.nan) == 'undefined'


def test_the_wall_is_controlled_by_its_flux_weighted_rate_control_not_the_dominant_path():
    """Pinholes carry 55 % and are 60 % surface-controlled; the intact path is 90 % metal.

    The dominant path's own step is the metal surface, but making metal diffusion faster raises
    the wall's flux most: X_metal = 0.55·0.4 + 0.45·0.9 = 0.625.
    """
    w = wall_from_paths(two_paths(0.55, {'oxide_diffusion': 0.1, 'metal_diffusion': 0.9},
                                  {'metal_surface': 0.6, 'metal_diffusion': 0.4}), 1.0)
    assert w['label'] == 'pinholes/metal diffusion'
    assert w[X_KEYS['metal diffusion']] == pytest.approx(0.625)
    assert sum(w[k] for k in X_KEYS.values()) == pytest.approx(1.0)


def test_control_needs_more_than_the_threshold():
    assert controlling(rate_control(oxide_diffusion=CONTROL, metal_diffusion=1 - CONTROL)) == MIXED_CONTROL
    assert controlling(rate_control(oxide_diffusion=CONTROL + 1e-9, metal_diffusion=1 - CONTROL - 1e-9)) == 'oxide diffusion'
    assert controlling(rate_control(oxide_diffusion=math.nan)) == 'undefined'


def test_mixed_defects_split_the_area_like_the_solver():
    fractions = defect_fractions(OXIDE_DEFECTS)
    assert math.isclose(sum(fractions.values()), OXIDE_DEFECTS['area_fraction'])
    components = OXIDE_DEFECTS['components']
    assert math.isclose(fractions['pinhole'] / fractions['crack'], components['pinholes'] / components['cracks'])
    assert defect_fractions({'type': 'crack', 'area_fraction': 0.2}) == {'crack': 0.2}
    assert defect_fractions({'type': 'mixed', 'area_fraction': 0.2, 'components': {}}) == {'pinhole': 0.2}


@pytest.mark.parametrize('T,P', POINTS)
@pytest.mark.parametrize('level', ['L3', 'L5'])
def test_model1_paths_add_up_to_the_wall_flux(level, T, P):
    """Re-solving each path alone must reproduce the wall the parallel-path solver assembled."""
    ox, me = layers(T)
    if level == 'L3':
        wall  = calculate_parallel_path_flux(P, SIM['P_downstream'], ox, me, OXIDE_DEFECTS)
        paths = model1_paths(level, P, SIM['P_downstream'], ox, me, OXIDE_DEFECTS)
    else:
        wall  = calculate_parallel_path_flux_defective_metal(P, SIM['P_downstream'], ox, me, OXIDE_DEFECTS, T, **MICRO_M1)
        paths = model1_paths(level, P, SIM['P_downstream'], ox, me, OXIDE_DEFECTS, T, MICRO_M1)
    assert set(paths) == {'intact', 'pinhole', 'crack', 'grain_boundary'}
    assert np.isclose(sum(d['contribution'] for d in paths.values()), wall['flux_total'], rtol=1e-9)
    assert np.isclose(paths['intact']['flux'], wall['flux_intact_per_area'], rtol=1e-9)


@pytest.mark.parametrize('T,P', POINTS)
@pytest.mark.parametrize('level', ['L3L6', 'L5L6'])
def test_each_surface_is_named_for_its_own_path(level, T, P):
    """Where the solver says 'surface', the label names the surface that path really has."""
    r = model2_wall(level, T, P, SIM['defect_config'])
    for kind, d in model2_paths(level, r).items():
        raw = r['intact_path'] if kind == 'intact' else r['defect_paths'][kind]['path_result']
        if raw['rate_limiting'] == 'surface':
            assert d['step'] == SURFACE_OF[kind]
    assert np.isclose(sum(d['contribution'] for d in model2_paths(level, r).values()), r['J_total'], rtol=1e-9)
    assert SURFACE_OF['pinhole'] == 'metal surface' and SURFACE_OF['intact'] == 'oxide surface'


def test_a_dropped_defect_path_is_undefined_not_defect_free():
    """calculate_mixed_defect_flux_L6 skips a path whose solve failed; the label must not hide that."""
    r = model2_wall('L3L6', *POINTS[0], {'pinhole': {'area_fraction': 0.01}})
    assert label_model2('L3L6', r, {'pinhole': {'area_fraction': 0.01}}) != 'undefined'
    r['defect_paths'].pop('pinhole')
    assert label_model2('L3L6', r, {'pinhole': {'area_fraction': 0.01}}) == 'undefined'


def test_model1_wall_label_refuses_paths_that_do_not_add_up():
    T, P = POINTS[0]
    ox, me = layers(T)
    wall = dict(calculate_parallel_path_flux(P, SIM['P_downstream'], ox, me, OXIDE_DEFECTS))
    wall['flux_total'] *= 1.01
    with pytest.raises(ValueError, match='add up'):
        label_model1('L3', wall, P, SIM['P_downstream'], ox, me, OXIDE_DEFECTS)


def test_drawing_marks_defect_paths_hollow_and_mixed_unfitted():
    assert describe('pinholes/metal surface')[3] == 'hollow'
    assert describe('intact/oxide surface')[3] == 'filled'
    assert describe('oxide surface')[3] == 'filled'
    assert describe(f'{NO_DOMINANT_PATH}/metal diffusion')[3] == 'filled'
    assert describe('pinholes/metal surface')[1] != describe('intact/oxide surface')[1]
    assert not fitted(MIXED_CONTROL) and not fitted(f'intact/{MIXED_CONTROL}') and not fitted('undefined')
    assert not fitted(f'{NO_DOMINANT_PATH}/metal diffusion')
    assert fitted('cracks/oxide diffusion') and fitted('metal diffusion')


# =============================================================================
# The whole-wall degrees of rate control are d ln J / d ln k of the wall
# =============================================================================

EPS = 1e-5


def drc(J, k):
    """d ln J / d ln k by central difference; J is a function of the scaled rate."""
    return (np.log(J(k * (1 + EPS))) - np.log(J(k * (1 - EPS)))) / (np.log(1 + EPS) - np.log(1 - EPS))


@pytest.mark.parametrize('T,P', POINTS)
def test_model1_wall_rate_control_is_the_derivative_of_the_wall_flux(T, P):
    """Oxide diffusion is perturbed on every oxide path at once (D_ox), metal diffusion on all paths."""
    ox, me = layers(T)
    P_down = SIM['P_downstream']
    wall = calculate_parallel_path_flux(P, P_down, ox, me, OXIDE_DEFECTS)
    w = wall_model1('L3', wall, P, P_down, ox, me, OXIDE_DEFECTS)
    X_ox = drc(lambda d: calculate_parallel_path_flux(P, P_down, {**ox, 'D_ox': d}, me, OXIDE_DEFECTS)['flux_total'],
               ox['D_ox'])
    X_m = drc(lambda d: calculate_parallel_path_flux(P, P_down, ox, {**me, 'D_metal': d}, OXIDE_DEFECTS)['flux_total'],
              me['D_metal'])
    assert w[X_KEYS['oxide diffusion']] == pytest.approx(X_ox, abs=1e-4)
    assert w[X_KEYS['metal diffusion']] == pytest.approx(X_m, abs=1e-4)
    assert sum(w[k] for k in X_KEYS.values()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('T,P', POINTS)
def test_model2_wall_rate_control_is_the_derivative_of_the_wall_flux(T, P):
    """Each surface is perturbed through k_diss at fixed K_eq, the convention of the solvers' fractions."""
    p  = get_all_properties(SIM['oxide_name'], SIM['metal_name'], T)
    sm = get_surface_kinetics_from_config(SIM['metal_name'], T, METALS)
    defects = SIM['defect_config']
    base = dict(k=p['k_diss'], km=sm['k_diss'], dox=p['D_ox'], dm=p['D_m'])

    def wall(**kw):
        a = {**base, **kw}
        return calculate_mixed_defect_flux_L6(P, SIM['P_downstream'], SIM['L_metal'], a['k'], p['K_eq'], a['dox'],
                                              p['K_ox'], SIM['L_oxide'], a['dm'], p['K_s_m'], defects,
                                              k_diss_metal=a['km'], K_eq_metal=sm['K_eq'])
    w = wall_model2('L3L6', wall(), defects)
    expected = {'oxide surface':   drc(lambda x: wall(k=x)['J_total'], base['k']),
                'metal surface':   drc(lambda x: wall(km=x)['J_total'], base['km']),
                'oxide diffusion': drc(lambda x: wall(dox=x)['J_total'], base['dox']),
                'metal diffusion': drc(lambda x: wall(dm=x)['J_total'], base['dm'])}
    for process, X in expected.items():
        assert w[X_KEYS[process]] == pytest.approx(X, abs=1e-4), process
    assert sum(w[k] for k in X_KEYS.values()) == pytest.approx(1.0, abs=1e-9)


def test_limits_bare_metal_and_intact_oxide():
    """f -> 0 is the intact path's own X; the pinhole path alone is wholly metal-controlled in Model 1."""
    T, P = POINTS[1]
    ox, me = layers(T)
    P_down = SIM['P_downstream']
    tiny = {'type': 'pinhole', 'area_fraction': 1e-15}
    wall = calculate_parallel_path_flux(P, P_down, ox, me, tiny)
    from calculations.interface_solver import calculate_oxide_metal_system
    alone = wall_model1('L2b', calculate_oxide_metal_system(P, P_down, ox, me))
    near_zero = wall_model1('L3', wall, P, P_down, ox, me, tiny)
    for key in X_KEYS.values():
        assert near_zero[key] == pytest.approx(alone[key], abs=1e-9)
    assert wall_model1('L1')[X_KEYS['metal diffusion']] == 1.0
