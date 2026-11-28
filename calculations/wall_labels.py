# calculations/wall_labels.py
r"""
Regime labels that say which path a wall's hydrogen takes and which process controls its rate.

A wall with a defective oxide is several paths side by side: the intact oxide, pinholes,
cracks and oxide grain boundaries. They exchange no hydrogen, so their fluxes add, and two
questions describe a point:

* **Which path dominates?** The one carrying more than half the wall's flux
  (:data:`DOMINANCE`). Parallel paths add, so the dominant path is the fastest route, not the
  slowest. If no path carries more than half, the point is ``mixed paths``. This sets the
  marker: filled for the intact oxide, hollow for a defect path.
* **Which process controls the rate?** The one with the largest degree of rate control
  (Campbell) for the *whole wall*, :math:`X_k = \partial\ln J/\partial\ln k`: how much the
  wall's flux rises when process *k* alone is made faster. It controls when
  :math:`X_k >` :data:`CONTROL`; otherwise the point is ``mixed control``. This sets the colour.

The processes are the gas-facing surfaces and the two layers: ``oxide surface`` (H2
dissociation on the oxide, on the intact area, under cracks and over oxide grain boundaries),
``metal surface`` (on the bare metal under pinholes, or a wall with no oxide), ``oxide
diffusion`` and ``metal diffusion``. Model 1 has no surface process: its surfaces are at
equilibrium.

The paths are independent, :math:`J = \sum_p a_p J_p` with area fractions :math:`a_p`, so the
wall's rate control is exactly the flux-weighted sum of each path's own,

.. math::

    X_k = \sum_p \frac{a_p J_p}{J}\,X_{k,p}

with :math:`X_{k,p} = 0` on a path that lacks process *k*. A process shared by several paths
(metal diffusion on all of them, oxide diffusion on the intact, crack and GB paths, the oxide
surface on the same three) is perturbed on all of them at once, so its terms add. Each path's
:math:`X_{k,p}` sum to one, so the wall's do too. The per-path values are the solvers' own:

* Model 1: :math:`X_{ox} = R_{ox}/(R_{ox}+R_m)` with :math:`R_m = 2L\sqrt{P_{int}}/(DK_s)`, the
  metal's differential resistance in pressure at the solved interface. This is
  :func:`calculations.sensitivity.series_rate_control_L5`, the split the L5 SA uses.
* Model 2: the solver's ``fraction_surface``, ``fraction_oxide`` and ``fraction_metal``
  (see :func:`calculations.surface_kinetics.surface_resistance`); the surface is perturbed
  through ``k_diss`` at fixed ``K_eq``.

Labels read ``<path>/<process>`` for a wall with defects, e.g. ``pinholes/metal surface``,
``intact/oxide diffusion`` or ``mixed paths/metal diffusion``, and ``<process>`` for a wall that
is a single path. ``undefined`` marks a point whose solve failed.

The area fraction of a defect is geometry, not a rate, and is not one of the processes.
"""

import numpy as np

from calculations.permeation_calc import calculate_simple_metal_flux
# calculations.interface_solver and calculations.defective_metal are imported inside model1_paths():
# they load the active study's config, and describe()/fitted() must import without one -- the study
# comparison runs outside every study.

#: A path dominates when it carries more than this share of the wall's flux.
DOMINANCE = 0.5

#: A process controls the wall's rate when its degree of rate control exceeds this.
CONTROL = 0.5

#: The rate processes, in display order.
PROCESSES = ('oxide surface', 'metal surface', 'oxide diffusion', 'metal diffusion')

#: Process -> the column holding its degree of rate control in a path or wall record.
X_KEYS = {p: 'X_' + p.replace(' ', '_') for p in PROCESSES}

#: Solver path key -> the path's name in a label.
PATH_NAMES = {'intact': 'intact', 'pinhole': 'pinholes', 'crack': 'cracks', 'grain_boundary': 'oxide GBs'}

#: Path part of a label when no path carries more than half the flux.
NO_DOMINANT_PATH = 'mixed paths'

#: Step part of a label when no process has X above CONTROL.
MIXED_CONTROL = 'mixed control'

#: The surface the gas meets on each path of a wall with defects.
SURFACE_OF = {'intact': 'oxide surface', 'pinhole': 'metal surface',
              'crack': 'oxide surface', 'grain_boundary': 'oxide surface'}

#: The surface of each Model 2 level that is a single path.
SINGLE_PATH_SURFACE = {'L1L6': 'metal surface', 'L2aL6': 'oxide surface',
                       'L2bL6': 'oxide surface', 'L4L6': 'oxide surface'}

_COMPONENT_KEY = {'pinhole': 'pinholes', 'crack': 'cracks', 'grain_boundary': 'grain_boundaries'}


# =============================================================================
# Building a label
# =============================================================================

def compose(step, path=None):
    """``'pinholes/metal surface'`` from its parts; ``path`` is a solver key, ``NO_DOMINANT_PATH`` or None."""
    head = [] if path is None else [PATH_NAMES.get(path, path)]
    return '/'.join(head + [step])


def parse(label):
    """``(path name or None, step)`` from a label: the inverse of :func:`compose`."""
    parts = label.split('/')
    path = parts.pop(0) if parts[0] in (*PATH_NAMES.values(), NO_DOMINANT_PATH) else None
    return path, parts[0]


def rate_control(**X):
    """A record of degrees of rate control, keyed by :data:`X_KEYS`; absent processes are 0."""
    return {X_KEYS[p]: float(X.get(p.replace(' ', '_'), 0.0)) for p in PROCESSES}


def controlling(record):
    """The process whose X exceeds :data:`CONTROL`, ``MIXED_CONTROL`` if none, ``'undefined'`` if X is NaN."""
    X = np.array([record[X_KEYS[p]] for p in PROCESSES], dtype=float)
    if not np.all(np.isfinite(X)):
        return 'undefined'
    i = int(np.argmax(X))
    return PROCESSES[i] if X[i] > CONTROL else MIXED_CONTROL


def wall_from_paths(paths, J_total):
    """The wall's record: label, dominant path, controlling process and whole-wall X.

    Parameters
    ----------
    paths : dict
        ``{solver path key: {'contribution', 'flux', 'step', X_KEYS...}}``, as returned by
        :func:`model1_paths` or :func:`model2_paths`.
    J_total : float
        The wall's flux, which the contributions add up to.

    Returns
    -------
    dict
        ``label``, ``dominant_path`` (a path name or ``NO_DOMINANT_PATH``), ``step`` and
        the wall's X, keyed by :data:`X_KEYS`. All NaN / ``'undefined'`` if the wall has no
        positive finite flux.
    """
    if not (np.isfinite(J_total) and J_total > 0):
        return undefined()
    share = {p: d['contribution'] / J_total for p, d in paths.items()}
    X = {key: sum(share[p] * paths[p][key] for p in paths) for key in X_KEYS.values()}
    step = controlling(X)
    if step == 'undefined':
        return undefined()
    top = max(share, key=share.get)
    path = top if share[top] > DOMINANCE else NO_DOMINANT_PATH
    return {'label': compose(step, path), 'dominant_path': PATH_NAMES.get(path, path), 'step': step, **X}


def single_path(X):
    """The record of a wall that is one path, from that path's X."""
    step = controlling(X)
    if step == 'undefined':
        return undefined()
    return {'label': step, 'dominant_path': None, 'step': step, **X}


def undefined():
    """The record of a point whose solve failed."""
    return {'label': 'undefined', 'dominant_path': None, 'step': 'undefined',
            **{key: np.nan for key in X_KEYS.values()}}


def label_from_paths(paths, J_total):
    """Label of a wall with defects; see :func:`wall_from_paths`."""
    return wall_from_paths(paths, J_total)['label']


# =============================================================================
# Model 1 (L1-L5)
# =============================================================================

def defect_fractions(defects):
    """Area fraction of each defect type, read the way the Level 3 solvers read an OXIDE_DEFECTS dict.

    A single type carries the whole ``area_fraction``. ``'mixed'`` splits it in proportion to
    ``components``; with no components the solvers treat the whole area as pinholes.
    """
    f = defects.get('area_fraction', 0.01)
    kind = defects.get('type', 'pinhole')
    if kind != 'mixed':
        return {kind: f}
    components = defects.get('components', {})
    total = sum(components.values())
    if total == 0:
        return {'pinhole': f}
    return {t: f * components[key] / total for t, key in _COMPONENT_KEY.items() if components.get(key, 0) > 0}


def model1_layer_X(r):
    """X of oxide and metal diffusion on one Model 1 oxide + metal path, from its solver result."""
    R_ox, R_m = r['R_oxide'], r['R_metal']
    return rate_control(oxide_diffusion=R_ox / (R_ox + R_m), metal_diffusion=R_m / (R_ox + R_m))


def model1_paths(level, P_up, P_down, oxide_props, metal_props, defects, temperature=None, micro=None):
    """Every path of a Model 1 wall with defects (L3 or L5), each solved on its own.

    The parallel-path solvers return the wall's flux and a label for the intact path only, so
    each path is re-solved here with the solver those modules use for it, with the same defaults:
    the intact, crack and grain-boundary paths with the oxide + metal system (thin oxide for a
    crack, faster oxide diffusion for a grain boundary), and a pinhole with the metal alone.
    Level 5 is Level 3 with the metal's diffusivity from
    :func:`calculations.defective_metal.metal_diffusivity`, exactly as
    :func:`calculations.parallel_oxide_defect_paths.calculate_parallel_path_flux_defective_metal`
    builds it.

    Parameters
    ----------
    level : {'L3', 'L5'}
    P_up, P_down : float
        Upstream and downstream pressure [Pa].
    oxide_props, metal_props : dict
        The layer properties the wall was solved with, ``thickness`` included; ``D_metal`` is
        the lattice diffusivity.
    defects : dict
        The OXIDE_DEFECTS-format dict the wall was solved with.
    temperature : float, optional
        [K]. L5 only.
    micro : dict, optional
        The keyword arguments the L5 solver was given: ``microstructure_params`` and,
        optionally, ``enable_gb_enhancement``. L5 only.

    Returns
    -------
    dict
        ``{path key: {'contribution', 'flux', 'step', X_KEYS...}}``: the path's area-weighted
        flux (the contributions add up to the wall's flux), its flux per unit area, the
        process controlling that path alone, and its degrees of rate control.
    """
    from calculations.interface_solver import calculate_oxide_metal_system
    if level == 'L5':
        from calculations.defective_metal import metal_diffusivity
        D_metal = metal_diffusivity(metal_props['D_metal'], temperature, **micro)
        metal_props = {**metal_props, 'D_metal': D_metal}

    def path(flux, area, X):
        return {'contribution': area * flux, 'flux': flux, 'step': controlling(X), **X}

    fractions = defect_fractions(defects)
    r = calculate_oxide_metal_system(P_up, P_down, oxide_props, metal_props)
    paths = {'intact': path(r['flux'], 1.0 - sum(fractions.values()), model1_layer_X(r))}
    for kind, area in fractions.items():
        if kind == 'pinhole':                                     # no oxide on this path: the metal alone
            r = calculate_simple_metal_flux(metal_props['D_metal'], metal_props['K_s_metal'],
                                            metal_props['thickness'], P_up, P_down)
            paths[kind] = path(r['flux'], area, rate_control(metal_diffusion=1.0))
            continue
        ox = dict(oxide_props)
        if kind == 'crack':
            ox['thickness'] = oxide_props['thickness'] * defects.get('thickness_factor', 0.1)
        else:
            ox['D_ox'] = oxide_props['D_ox'] * defects.get('diffusivity_factor', 10)
        r = calculate_oxide_metal_system(P_up, P_down, ox, metal_props)
        paths[kind] = path(r['flux'], area, model1_layer_X(r))
    return paths


def wall_model1(level, result=None, P_up=None, P_down=None, oxide_props=None, metal_props=None,
                defects=None, temperature=None, micro=None):
    """The record (label, dominant path, controlling process, X) of one Model 1 point.

    L1, L2a and L4 need no result: a bare metal (perfect or polycrystalline) is wholly
    controlled by metal diffusion and an oxide alone by oxide diffusion. L2b is one path, read
    from its solver's result. L3 and L5 re-solve each path (:func:`model1_paths`, which takes
    the remaining arguments) and raise if the paths do not add up to the wall flux the
    parallel-path solver returned in ``result``.
    """
    if level in ('L1', 'L4'):
        return single_path(rate_control(metal_diffusion=1.0))
    if level == 'L2a':
        return single_path(rate_control(oxide_diffusion=1.0))
    if level == 'L2b':
        return single_path(model1_layer_X(result))
    paths = model1_paths(level, P_up, P_down, oxide_props, metal_props, defects, temperature, micro)
    J_total = result['flux_total']
    J_paths = sum(d['contribution'] for d in paths.values())
    if not np.isclose(J_paths, J_total, rtol=1e-6, atol=0):
        raise ValueError(f'{level}: the paths add up to {J_paths:.6e}, the wall solver gave {J_total:.6e}')
    return wall_from_paths(paths, J_total)


def label_model1(*args, **kwargs):
    """Label of one Model 1 point; takes the arguments of :func:`wall_model1`."""
    return wall_model1(*args, **kwargs)['label']


# =============================================================================
# Model 2 (the L6 family)
# =============================================================================

def model2_X(path_result, surface):
    """X of one Model 2 path from the solver's fractions, its surface named for that path's surface."""
    res = path_result.get('resistances') or {}
    X = {X_KEYS[p]: 0.0 for p in PROCESSES}
    X[X_KEYS[surface]] = res.get('fraction_surface', np.nan)
    X[X_KEYS['oxide diffusion']] = res.get('fraction_oxide', 0.0)
    X[X_KEYS['metal diffusion']] = res.get('fraction_metal', 0.0)
    if not np.isfinite(path_result.get('flux', path_result.get('J_ss', np.nan))):
        X = {key: np.nan for key in X}
    return {key: float(v) for key, v in X.items()}


def model2_paths(level, result):
    """Every path of a Model 2 wall with defects (L3+L6 or L5+L6), from the solver's own breakdown.

    Returns the same structure as :func:`model1_paths`.
    """
    def path(path_result, surface, contribution):
        X = model2_X(path_result, surface)
        return {'contribution': contribution, 'flux': path_result.get('flux', np.nan),
                'step': controlling(X), **X}

    paths = {'intact': path(result['intact_path'], SURFACE_OF['intact'],
                            result['flux_breakdown']['intact']['contribution'])}
    for kind, d in result.get('defect_paths', {}).items():
        paths[kind] = path(d['path_result'], SURFACE_OF[kind], d['flux_contribution'])
    return paths


def wall_model2(level, result, defects=None):
    """The record (label, dominant path, controlling process, X) of one Model 2 point.

    For L3+L6 and L5+L6 pass the ``defect_config`` the wall was solved with: the L3+L6 solver
    drops a defect path whose solve failed, which would make the point look defect-free, so a
    point missing a requested path is ``undefined`` instead.
    """
    if level in ('L3L6', 'L5L6'):
        J_total = result.get('J_total', np.nan)
        requested = [k for k, cfg in (defects or {}).items() if cfg.get('area_fraction', 0) > 0]
        if any(k not in result.get('defect_paths', {}) for k in requested) or not np.isfinite(J_total):
            return undefined()
        return wall_from_paths(model2_paths(level, result), J_total)
    return single_path(model2_X(result, SINGLE_PATH_SURFACE[level]))


def label_model2(level, result, defects=None):
    """Label of one Model 2 point; see :func:`wall_model2`."""
    return wall_model2(level, result, defects)['label']


# =============================================================================
# Drawing a label
# =============================================================================

#: Controlling process -> (words, colour, marker). Colour means the process on every figure.
STEP_STYLE = {
    'oxide surface':   ('oxide-surface controlled (H$_2$ dissociation on the oxide)',      '#1baf7a', 'P'),
    'metal surface':   ('metal-surface controlled (H$_2$ dissociation on the bare metal)', '#8e5cc4', 'P'),
    'oxide diffusion': ('oxide-diffusion controlled',                                     '#eb6834', 's'),
    'metal diffusion': ('metal-diffusion controlled',                                     '#2a78d6', 'o'),
    MIXED_CONTROL:     ('mixed control, no process with X > 0.5 (not fitted)',            '#898781', 'X'),
    'undefined':       ('undefined (solve failed, not fitted)',                           '#898781', 'x'),
}
_DOMINANT_WORDS = {'intact': 'Intact oxide dominant', 'pinholes': 'Pinholes dominant',
                   'cracks': 'Cracks dominant', 'oxide GBs': 'Oxide GBs dominant',
                   NO_DOMINANT_PATH: 'No dominant path (not fitted)'}


def describe(label):
    """``(legend name, colour, marker, fill)`` for one label.

    Colour and marker are the controlling process, and a hollow marker means a defect path
    (pinholes, cracks or oxide grain boundaries) carries most of the flux.
    """
    path, step = parse(label)
    words, colour, marker = STEP_STYLE.get(step, (step, '#52514e', 'o'))
    name = f'{_DOMINANT_WORDS[path]}: {words}' if path else words[0].upper() + words[1:]
    return name, colour, marker, ('filled' if path in (None, 'intact', NO_DOMINANT_PATH) else 'hollow')


def fitted(label):
    """Points with no controlling process, no dominant path, or a failed solve are drawn but not fitted.

    A fit needs a window with one mechanism; where no path carries the majority the flux is the
    crossover between two, and not a power law in anything.
    """
    path, step = parse(label)
    return step not in (MIXED_CONTROL, 'undefined') and path != NO_DOMINANT_PATH
